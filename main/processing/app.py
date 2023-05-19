import numpy as np
import torch
import torch.nn as nn
from torch import sigmoid, tanh
import torch.optim as optim
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore,storage
from flask import Flask, request, jsonify
from datetime import datetime
import joblib
import random
import string
import pytz
import itertools
import re
import copy
from datetime import timedelta
import io
import time
import requests
from sys import getsizeof

cred_obj = credentials.Certificate('my-test-project-373716-firebase-adminsdk-nyuqz-bc57b26b33.json')

default_app = firebase_admin.initialize_app(cred_obj)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bucket = storage.bucket('my-test-project-373716.appspot.com')

db = firestore.client()

sched_ref = db.collection('schedule')
warn_ref = db.collection('warning') #reference to warning
machine_ticket_ref=db.collection('ticket_fromMachine')
desc_mai_ref=db.collection('maintenance_description')
app=Flask(__name__)


'''
Langkah kode processing sampai ticket:
1. Membuat fungsi untuk trigger-able processing. Trigger dilakukan oleh mini PC setiap kali data diterima (ini di akhir krn ada tambahan kode trigger).
2. Membuat fungsi untuk mengambil file model dari storage firebase (bucket).
3. Membuat kode program untuk memanggil fungsi predict di dalam Flask.
4. Membuat kode program untuk menyimpan keluaran dari fungsi predict tersebut ke dalam variabel.
5. Membuat kode untuk menyimpan variabel keluaran fungsi predict ke dalam firestore -> /processed.
'''
"""
Langkah-langkah processing scheduling
1. terima data schedule dari mesin berdasarkan input user
2. ambil data jadwal dari database firestore
3. Alokasikan machine_id==machine id dari jadwal ticketing dari mesin.
3. dapatkan start dan end datetime dari data yang telah dialokasikan.
"""
def rpm_process(label,data):
	message=""
	title=[]
	# if(data['stability']=="Unstable"):
	# 	message=message+"{} tidak stabil, di {}".format(label,str(round(data['value'][0],2)))
	# 	title.append(label + " tidak stabil")
	# 	priority_1=2
	# elif(data['stability']=="Stable"):
	# 	message=message+"{} stabil, di {}".format(label,str(round(data['value'][0],2)))
	# 	priority_1=0

	priority_2={'message':[],'priority':[]}

	if("Normal" not in data['stat']):
		message=message+"{} sangat tidak stabil dengan nilai {}".format(label,str(round(data['value'][0],2)))
		priority_2['priority']=[2,2,2]

	else:
		for i in (data['stat']):
			if(i=="Higher"):
				priority_2['message'].append('{} lebih tinggi dari batas toleransi 5% dengan nilai {}'.format(label,str(round(data['value'][0],2))))
				priority_2['priority'].append(2)
				title.append(label + " tidak stabil")
			elif(i=="Lower"):
				priority_2['message'].append('{} lebih rendah dari batas toleransi 5% dengan nilai {}'.format(label,str(round(data['value'][0],2))))
				priority_2['priority'].append(2)
				title.append(label + " tidak stabil")
			else:
				priority_2['message'].append('{} normal dengan nilai {}'.format(label,str(round(data['value'][0],2))))
				priority_2['priority'].append(0)

		message=message+priority_2['message'][priority_2['priority'].index(max(priority_2['priority']))]
	message=message+", Maksimum: {}, Minimum: {}".format(round(data['value'][1],2),round(data['value'][2],2))
	message=message+", Stabilitas: "+str(round(data['cv'],2))

	result={
		"message":message,
		"priority":priority_2['priority'],
		"title":title
	}
	# print("result RPM: ",result)
	print("memori RPM: ",getsizeof(result))
	return result

def temperature(label,data):
	message=""
	prior=-1
	
	prior_1={'message':[],'priority':[]}
	title=[]
	for i in (data['status']):
		if(i=="Danger"):
			prior_1['message'].append("{} bahaya: ".format(label)+ str(round(data['value'][0],2)))
			prior_1['priority'].append(2)
			title.append(label + " tinggi")
		elif(i=="High"):
			prior_1['message'].append("{} tinggi: ".format(label)+ str(round(data['value'][0],2)))
			prior_1['priority'].append(1)
		elif(i=="Warm"):
			prior_1['message'].append("{} sedang: ".format(label)+ str(round(data['value'][0],2)))
			prior_1['priority'].append(0)
		elif(i=="Normal"):
			prior_1['message'].append("{} normal".format(label))
			prior_1['priority'].append(0)
	
	message+=prior_1['message'][prior_1['priority'].index(max(prior_1['priority']))]

	message=message+", maksimum: {}, minimum: {}".format(str(round(data['value'][1],2)),str(round(data['value'][2],2)))
	if(data['trend']>0):
		trend="meningkat"
	elif(data['trend']<0):
		trend='menurun'
	else:
		trend='stabil'

	message = message + ", {} dengan gradien {}".format(trend,round(data['trend'],2))
	result={
		"message":message,
		"priority":prior_1['priority'],
		"title":title
	}
	print("memori temperature: ",getsizeof(result))
	return result

"""Memanggil Fungsi Predict di dalam task"""
def predict_pipeline(label,data,value,machine_id):
	try:
		ppp=time.time()
		params = db.collection('processing_params').document(machine_id).get().to_dict()
		# TIME_STEPS = params['TIME_STEPS']
		# NUM_UNITS = params['NUM_UNITS']
		# LEARNING_RATE = params['LEARNING_RATE']
		title=[]
		params['data']=data
		priority_2=[]

		# print("params: ",params)
		resp=requests.request("POST",params['METHOD'],json=params)
		# print("respond parameter: ",resp.json())
		print("elapsed time prediction: ",resp.elapsed.total_seconds())
		# print(resp.items())
		resp=resp.json()
		try:
			first,message_1=resp['1']
			second,message_2=resp['2']
			third,message_3=resp['3']
			sorted_list=resp['sorted_list']

			if(first!=0):
				message="Prediksi {} Bahaya (".format(label)+message_1+" "+str(round(sorted_list[0]*100,2))+"%, "+message_2+" "+str(round(sorted_list[1]*100,2))+"%, "+message_3+" "+str(round(sorted_list[2]*100,2))+"%)"
				title.append(label + " {}".format(message_1))
				priority_2.append(2)
			else:
				message="Prediksi {} Baik (".format(label)+message_1+" "+str(round(sorted_list[0]*100,2))+"%, "+message_2+" "+str(round(sorted_list[1]*100,2))+"%, "+message_3+" "+str(round(sorted_list[2]*100,2))+"%)"
				priority_2.append(0)

		except Exception as e:
			message="Prediksi: Error ketika melakukan prediksi jenis kerusakan. Error: {}. ".format(str(resp))
			print(message)
			# priority_1=0

		if(value[0]=="Danger"):
			title.append("{} {}".format(label,"tidak stabil"))
			priority_2.append(2)
		elif(value[0]=="Alert"):
			title.append("{} {}".format(label,"tidak stabil"))
			priority_2.append(2)
		elif(value[0]=='Satisfactory'):
			priority_2.append(0)
		elif(value[0]=='Good'):
			priority_2.append(0)
		
		# print({
		# 	'message':"Vrms {}: {} m/sRMS".format(value[0],round(value[1],2)),
		# 	'priority':priority_2[-1]
		# })
		message=message+", Vrms {}: {} m/sRMS".format(value[0],round(value[1],2))

		# priority=[priority_1,priority_2]
		# print(priority_2)
		result={
			"message":message,
			"priority":priority_2,
			"title":title
		}
		# print("result vibrasi: ",result)
		print("memori vibrasi: ",getsizeof(result))
		return result
	
	except Exception as e:
		return jsonify(e),400

def enterpret(data):
	if(data==0):
		return 'Normal'
	elif(data==1):
		return "Imbalance"
	elif(data==2):
		return "Horizontal Misalignment"
	elif(data==3):
		return 'Vertical Misalignment'
	# elif(data==4):
	# 	return "Horizonal Misalignment Mild"
	# elif(data==5):
	# 	return "Horizonal Misalignment Medium"
	# elif(data==6):
	# 	return "Horizonal Misalignment High"
	# elif(data==7):
	# 	return "Horizonal Misalignment Danger"
	# elif(data==8):
	# 	return "Vertical Misalignment High"
	# elif(data==9):
	# 	return "Vertical Misalignment Medium"
	# elif(data==10):
	# 	return "Vertical Misalignment Low"

#utilities
class JANet(nn.Module):
	def __init__(self, inputs, cells, num_outputs, num_timesteps, output_activation=None):
		super(JANet, self).__init__()
		
		self.inputs = inputs
		self.cells = cells
		self.classes = num_outputs
		self.num_timesteps = num_timesteps
		self.output_activation = output_activation
		
		kernel_data = torch.zeros(inputs, 2 * cells, dtype=torch.get_default_dtype())
		kernel_data = torch.nn.init.xavier_uniform_(kernel_data)
		self.kernel = nn.Parameter(kernel_data)
		
		recurrent_kernel_data = torch.zeros(cells, 2 * cells, dtype=torch.get_default_dtype())
		recurrent_kernel_data = torch.nn.init.xavier_uniform_(recurrent_kernel_data)
		self.recurrent_kernel = nn.Parameter(recurrent_kernel_data)
		
		recurrent_bias = np.zeros(2 * cells)
		# chrono initializer
		recurrent_bias[:cells] = np.log(np.random.uniform(1., self.num_timesteps - 1, size=cells))
		recurrent_bias = recurrent_bias.astype('float32')
		self.recurrent_bias = nn.Parameter(torch.from_numpy(recurrent_bias))
		
		self.output_dense = nn.Linear(cells, num_outputs)
		
	def forward(self, inputs):
		h_state = torch.zeros(inputs.size(0), self.cells, dtype=torch.get_default_dtype()).to(device)
		c_state = torch.zeros(inputs.size(0), self.cells, dtype=torch.get_default_dtype()).to(device)
		
		num_timesteps = inputs.size(1)
		
		for t in range(num_timesteps):
			ip = inputs[:, t, :]
			
			z = torch.mm(ip, self.kernel)
			z += torch.mm(h_state, self.recurrent_kernel) + self.recurrent_bias
			
			z0 = z[:, :self.cells]
			z1 = z[:, self.cells: self.cells * 2]
			
			f = sigmoid(z0)
			c = f * c_state + (1. - f) * tanh(z1)
			
			h = c
			
			h_state = h
			c_state = c
		
		preds = self.output_dense(h_state)
		
		if self.output_activation is not None:
			preds = self.output_activation(preds)
		
		return preds

def processing_schedule_mai(percentages,data,sched):
	temp={}
	for i in sched:
		if((data['StartTime']>sched[i]['StartTime'])):
			"""
			Jika start produksi lebih dulu dibanding
			start maintenance
			"""
			if(data['StartTime']<=sched[i]['EndTime']):
				data['StartTime']=sched[i]['EndTime']
				data['EndTime']=data['StartTime']+data['duration']
				print("hoho 4 ",data['StartTime'],data['EndTime'])

		elif((data['StartTime']<=sched[i]['StartTime'])):
			"""Jika start time maintenance lebih dulu dibanding
			start time produksi"""
			if(data['EndTime']<=sched[i]['StartTime']):
				print("hoho hoho 3 ",data['StartTime'],data['EndTime'])
				"""Jika end time dari maintenance lebih dulu
				dibanding start time produksi"""
				pass

			if((data['EndTime']<=sched[i]['EndTime']) and (data['EndTime']>=sched[i]['StartTime'])):
				"""Jika end time dari maintenance lebih dulu
				dibanding end time dari end time produksi"""
				if((sched[i]['EndTime']-data['EndTime'])<sched[i]['duration']):
					"""jika selisih """
					temp[i]=data['EndTime']-sched[i]['StartTime']
					sched[i]['StartTime']=data['EndTime']
					sched[i]['EndTime']=sched[i]['StartTime']+sched[i]['duration']
					print(data)
					print("print lagi ",data['StartTime'],data['EndTime'])
					print(i,sched[i]['StartTime'],sched[i]['EndTime'])
					post_processing_mai(temp,sched)
					temp.pop(i)
				else:
					"""perlu ada ngecek priority/urgensi dari ticketing"""
					if(percentages[0]>50 or percentages[1]>50 or percentages[2]>50 or percentages[3]>50):
						data['StartTime']=sched[i]['EndTime']
						data['EndTime']=data['StartTime']+data['duration']
						print("lah kok ",data['StartTime'],data['EndTime'])
					else:
						temp[i]=data['EndTime']-sched[i]['StartTime']
						sched[i]['StartTime']=data['EndTime']
						sched[i]['EndTime']=sched[i]['StartTime']+sched[i]['duration']
						print(data)
						print("print lagi ",data['StartTime'],data['EndTime'])
						print(i,sched[i]['StartTime'],sched[i]['EndTime'])
						post_processing_mai(temp,sched)
						temp.pop(i)

			elif(data['EndTime']>=sched[i]['EndTime']):
				"""Harus cek tingkat kerusakan warning dari mesin
				Sementara kita mutlak utamakan maintenance dibanding produksi

				Dengan kata lain, produksi mundur
				"""
				if(percentages[0]>50 or percentages[1]>50 or percentages[2]>50 or percentages[3]>50):
					temp[i]=data['EndTime']-sched[i]['StartTime']
					print(i,sched[i]['StartTime'],sched[i]['EndTime'])
					sched[i]['StartTime']=data['EndTime']
					sched[i]['EndTime']=sched[i]['StartTime']+sched[i]['duration']
					print('kenapa ini ',data['StartTime'],data['EndTime'])
					print(i,sched[i]['StartTime'],sched[i]['EndTime'])
					post_processing_mai(temp,sched)
					print(i,sched[i]['StartTime'],sched[i]['EndTime'])
					temp.pop(i)
				else:
					data['StartTime']=sched[i]['EndTime']
					data['EndTime']=data['StartTime']+data['duration']		

		# else: #start time mai==start time produksi
		# 	if(data['duration']<=sched[i]['duration']):
		# 		sched[i]['StartTime']=data['EndTime']
		# 		sched[i]['EndTime']=sched[i]['StartTime']+sched[i]['duration']
		# 		temp[i]=sched[i]['duration']
		# 		post_processing_mai(temp,sched)
		# 		temp.pop(i)
		# 	else:
		# 		data['StartTime']=sched[i]['EndTime']
		# 		data['EndTime']=data['StartTime']+data['duration']
		
		# print(sched[i]['StartTime'],sched[i]['EndTime'])

	# print(sched)
	# print("Selesai bang")
	# print('nilai temp: ',temp)

def post_processing_mai(temp,sched):
	key_temp=list(temp.keys())
	sched_key=list(sched.keys())
	# print('temp: ',temp)
	# print('key temp: ',key_temp)
	# print('sched key awal: ',sched_key)

	# for i in range(len(sched_key)):
	# 	print(i,sched_key[i])

	tempp=copy.deepcopy(sched_key)
	for i in key_temp:
		for key,j in enumerate(tempp):
			if(j!=i):
				sched_key.remove(j)
			else:
				break

	for i in sched:
		for j in sched:
			if(i!=j and j in sched_key):
				if((sched[i]['StartTime']<=sched[j]['StartTime'] and sched[i]['EndTime']>=sched[j]['StartTime'])
	   				or (sched[i]['EndTime']>sched[j]['EndTime'])):
					# print('ijeka ',i,j,sched[j]['StartTime'],sched[j]['EndTime'])
					for k in temp:
						if(j!=k):
							# print(i,j,k) 
							# print("awal: ")
							# print(j,k,sched[j]['StartTime'],sched[j]['EndTime'])
							sched[j]['StartTime']+=temp[k]
							sched[j]['EndTime']+=temp[k]
							# print('akhir: ')
							# print(j,k,sched[j]['StartTime'],sched[j]['EndTime'])
						else:
							# try:
							# 	print('hah: ',j,k,sched[j]['StartTime'],sched[j]['EndTime'])
							# 	sched[j]['StartTime']-=temp[k]
							# 	sched[j]['EndTime']-=temp[k]
							# except:
							# 	break
							break
					
					sched_key.remove(j)

def generate_schedule():
	data={
	"2023-19-04@12:12:36-Produksi":
	{
		"StartTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 7, 0)),
		"User": "Produksi",
		"Description": "Bulk02",
		"EndTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 10, 59)),
		"Id": "2023-19-04@12:12:36-Produksi",
		"Subject": "LHL01"},
	"2023-19-04@12:13:03-Baejah":
	{
		"StartTime": pytz.timezone('UTC').localize(datetime(2023, 4, 19, 7, 0)),
		"User": "Baejah",
		"Description": "Cek Mesin",
		"EndTime": pytz.timezone('UTC').localize(datetime(2023, 4, 19, 10, 59)),
		"Id": "2023-19-04@12:13:03-Baejah",
		"Subject": "LHL01"},
	"2023-19-04@12:12:37-Produksi":
	{
		"StartTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 12, 0)),
		"User": "Produksi",
		"Description": "Bulk02",
		"EndTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 12, 59)),
		"Id": "2023-19-04@12:12:37-Produksi",
		"Subject": "LHL01"},
	"2023-19-04@12:12:38-Produksi":
	{
		"StartTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 15, 0)),
		"User": "Produksi",
		"Description": "Bulk02",
		"EndTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 21, 59)),
		"Id": "2023-19-04@12:12:38-Produksi",
		"Subject": "LHL01"},
	"2023-19-04@12:12:39-Produksi":
	{
		"StartTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 17, 0)),
		"User": "Produksi",
		"Description": "Bulk02",
		"EndTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 20, 59)),
		"Id": "2023-19-04@12:12:38-Produksi",
		"Subject": "LHL01"},
	"2023-19-04@12:12:40-Produksi":
	{
		"StartTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 21, 0)),
		"User": "Produksi",
		"Description": "Bulk02",
		"EndTime": pytz.timezone('UTC').localize(datetime(2023, 4, 18, 23, 59)),
		"Id": "2023-19-04@12:12:38-Produksi",
		"Subject": "LHL01"},
	}
	return data

def json_data():
	data={'EndTime': pytz.timezone('UTC').localize(datetime(2023, 4, 18, 16, 15)),
		'StartTime': pytz.timezone('UTC').localize(datetime(2023, 4, 18, 10, 0)),
		'Subject': 'LHL01',
		'Description': 'cek mesin status',
		'Id': '123bange',
		'User': 'None'}
	return data

def read_from_firestore(data):
	tm=datetime.now()
	schedule=sched_ref.where('Subject','==',data['Subject']).stream()

	sched = {}
	for doc in schedule:
		sched[doc.to_dict()['Id']] = doc.to_dict()

	sched_key=list(sched.keys())
	for i in sched_key:
		if(sched[i]['EndTime']<data['StartTime']):
			sched.pop(i)
	latency=datetime.now()-tm
	return sched,latency

def sorting_sched(sched):
	key_pair = list(sched.items())
	key_pair.sort(key=lambda x: x[1]['StartTime'])
	sched=dict(key_pair)

	return sched

def to_firestore(method,data):
	if(method=='send'):
		sched_ref.document(data['Id']).set(data)
	elif(method=='update'):
		sched_ref.document(data['Id']).update({
					'StartTime':data['StartTime'],
					'EndTime':data['EndTime']})

def duration_sched(urgency):
	if(urgency>50):
		duration=timedelta(hours=4)
	else:
		duration=timedelta(hours=2)
	return duration

def warning_to_schedule(percentages,data):
	"""Ambil urgensi dari string details"""
	# urgency=float(re.findall(r'\d+\.\d+',data['detail'])[0])

	if(percentages[0]>50 or percentages[1]>50 or percentages[2]>50 or percentages[3]>50):
		duration=timedelta(hours=4)
	else:
		duration=timedelta(hours=2)
	
	StartTime=data['created']
	schedule={
		'StartTime':StartTime,
		'EndTime':StartTime+duration,
		'Subject': data['machine_id'],
		'Description':data['title'],
		'Id':datetime.now(pytz.timezone('Asia/Bangkok')).strftime("%Y-%d-%m@%H:%M:%S")+'-Machine',
		'User':'Machine'
	}

	return schedule

def json_data_ticket():
	data_for_ticket={
			'id':datetime.now().strftime("%Y-%d-%m@%H:%M:%S")+"-Machine-"+"LHL01",
			'title':"Masalah Temperatur",
			'detail':"85.00% rusak\\nTemperatur tinggi, bernilai 100 Celcius",
			'issuer':"Machine",
			'machine_id':"LHL01",
			'plant':"J2",
			'status':1,
			'created':datetime.now(pytz.timezone('Asia/Bangkok')),
			'deadline':datetime.now(pytz.timezone('Asia/Bangkok')),
			'modified':datetime.now(pytz.timezone('Asia/Bangkok'))}
	
	return data_for_ticket

"""Triggerable processing"""
@app.route("/warning-machine",methods=["GET","POST"])
def processing():
	tm=datetime.now()
	if(request.method=='POST'):
		try:   
			data=request.json

		except KeyError:
			return jsonify({'error': "No text sent"})

		try:
			date=data['time']
			line=data['line']
			plant=data['plant']
			zona=data['zona']
			deviceName=data['machine_id']

			message={}
			priority_list=[]
			priority_temp=[]
			priority_rpm=[]
			priority_vib=[]
			key_data=list(data.keys())
			priority_index=[i for i in key_data if re.search("(temperature)|(vibration)|(rpm)",i)]

			for i in key_data:
				ppp=time.time()
				if(re.search('temperature',i)):
					message[i]=temperature(i,data[i])
					print("Temperature: ",time.time()-ppp)

				elif(re.search('rpm',i)):
					message[i]=rpm_process(i,data[i])
					print("RPM: ",time.time()-ppp)
				
				elif(re.search('vibration',i)):
					message[i]=predict_pipeline(i,data[i]['wavelet'],[data[i]['rms_category'],data[i]['rms_value']],deviceName)
					print("Vibration: ",time.time()-ppp)

			ppp_1=time.time()
			for key in priority_index:
				priority_list.append(message[key]['priority'])
			
			for key in priority_index:
				if(re.search('temperature',key)):
					priority_temp.append(message[key]['priority'])
				elif(re.search('rpm',key)):
					priority_rpm.append(message[key]['priority'])
				elif(re.search('vibration',key)):
					priority_vib.append(message[key]['priority'])

			flat_list=list(itertools.chain(*priority_list))
			temp_list=list(itertools.chain(*priority_temp))
			rpm_list=list(itertools.chain(*priority_rpm))
			vib_list=list(itertools.chain(*priority_vib))
			message_temp="{} dari {} ({}%): ".format(len([1 for i in temp_list if i==2]),
					 len(temp_list),round(len([1 for i in temp_list if i==2])*100/len(temp_list),2))
			
			message_rpm="{} dari {} ({}%): ".format(len([1 for i in rpm_list if i==2]),
					 len(rpm_list),round(len([1 for i in rpm_list if i==2])*100/len(rpm_list),2))
			
			message_vib="{} dari {} ({}%): ".format(len([1 for i in vib_list if i==2]),
					 len(vib_list),round(len([1 for i in vib_list if i==2])*100/len(vib_list),2))
			
			priority=max(flat_list)
			if(priority==2):
				try:
					unique_title="Cek {}".format(", ".join(list(set(list(itertools.chain(*[message[key]['title'] for key in priority_index]))))))
				except Exception as e:
					print("Error unique title: ",e)
					return jsonify(e),400
				# print("title: ",unique_title)
				urgency=len([1 for i in flat_list if i==2])/len(flat_list)
				print('urgency: ',urgency)
				message_temp=message_temp+";  ".join([message[k]['message'] for k in priority_index if re.search('temperature',k)])
				message_rpm=message_rpm+";  ".join([message[k]['message'] for k in priority_index if re.search('rpm',k)])
				message_vib=message_vib+";  ".join([message[k]['message'] for k in priority_index if re.search('vibration',k)])

				details='''{} dari {} ({}%) Indikator terdeteksi abnormal pada {} dengan detail: {}.     {}.     {}'''.format(
					len([1 for i in flat_list if i==2]),len(flat_list),
					round(urgency*100,2),date,message_temp,
												message_rpm,
												message_vib)

				try:          
					if(urgency>=0.75):
						deadline=datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=pytz.timezone("GMT"))+timedelta(hours=8)
					elif(urgency>=0.5):
						deadline=datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=pytz.timezone("GMT"))+timedelta(hours=16)
					else:
						deadline=datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=pytz.timezone("GMT"))+timedelta(days=1)

					"""ambil data terakhir dari firestore"""
					print("bangg")

					data_for_ticket={
						'id':datetime.now(pytz.timezone('Asia/Bangkok')).strftime("%Y-%d-%m@%H:%M:%S") + "-Machine-"+deviceName,
						'title':unique_title,
						'detail':details,
						'issuer':"Machine",
						'machine_id':deviceName,
						'plant':plant,
						'status':1,
						'created':datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=pytz.timezone("GMT")),
						'deadline':deadline,
						'modified':datetime.now(pytz.timezone('Asia/Bangkok')).replace(tzinfo=pytz.timezone("GMT"))}

					ppp=db.collection('ticket_fromMachine').where('machine_id','==',deviceName).where('issuer','==','Machine').stream()
					pp=[]				
					for p in ppp:
						pp.append(p.to_dict())

					if(len(pp)>0):
						pp.sort(key=lambda pp: pp['created'])
						temp=pp[-1]
						print("hh")
						keys=[float(i) for i in re.findall(r'\((\d+\.\d+)', temp['detail'])]

						if(keys[1]!=round(len([1 for i in temp_list if i==2])*100/len(temp_list),2) or 
							keys[2]!=round(len([1 for i in rpm_list if i==2])*100/len(rpm_list),2) or
							keys[3]!=round(len([1 for i in vib_list if i==2])*100/len(vib_list),2) or
							keys[0]!=round(urgency*100,2)):
							
							print("memori data tiket: ",getsizeof(data_for_ticket))
							# print("ticket: ",data_for_ticket)
							# print(data_for_ticket)
							print("processing sisanya: ",time.time()-ppp_1)
							# print('Ticket: ',data_for_ticket)
							"""kirim data ke database bagian ticketing"""
							kkk=datetime.now()
							print("Successfully processed with execution time " + str(datetime.now()-tm))

							machine_ticket_ref.document(data_for_ticket['id']).set(data_for_ticket, merge=True)
							print("Latency of sending: ",datetime.now()-kkk)
							print("Successfully sent with execution time " + str(datetime.now()-tm))

							return jsonify(message=message, data=data), 200	

						else:
							return jsonify("ticket same"),200	
					else:
						print('ticket baru')
						print("memori data tiket: ",getsizeof(data_for_ticket))
						# print("ticket: ",data_for_ticket)
						# print(data_for_ticket)
						print("processing sisanya: ",time.time()-ppp_1)
						# print('Ticket: ',data_for_ticket)
						"""kirim data ke database bagian ticketing"""
						kkk=datetime.now()
						print("Successfully processed with execution time " + str(datetime.now()-tm))

						machine_ticket_ref.document(data_for_ticket['id']).set(data_for_ticket, merge=True)
						print("Latency of sending: ",datetime.now()-kkk)
						print("Successfully sent with execution time " + str(datetime.now()-tm))

						return jsonify(message=message, data=data), 200										
				except Exception as e:
					return jsonify(e),400
			else:
				return jsonify(message="No warning detected :)"), 200
			
		except Exception as e:   
			return jsonify({'error': str(e)}) 
		
@app.route('/schedule-processing',methods=['POST'])
def sched_processing():
	data=None
	try:
		temp=time.time()
		temp_time=datetime.now()
		ticket=request.json
		try:
			ticket['created']=datetime.strptime(ticket['created'],"%a %b %d %Y %H:%M:%S GMT%z (Coordinated Universal Time)")
			ticket['modified']=datetime.strptime(ticket['modified'],"%a %b %d %Y %H:%M:%S GMT%z (Coordinated Universal Time)")
			ticket['deadline']=datetime.strptime(ticket['deadline'],"%a %b %d %Y %H:%M:%S GMT%z (Coordinated Universal Time)")

		except Exception as e:
			message='error defining datetime: '+ str(e)
			print(message)
			return jsonify(message=message),400

		percentages = [float(i) for i in re.findall(r'\((\d+\.\d+)', ticket['detail'])]

		data=warning_to_schedule(percentages,ticket)
		print("memori data mai: ",getsizeof(data))
		data['duration']=data['EndTime']-data['StartTime']
		print("waktu eksekusi konversi jadi jadwal mai: ",time.time()-temp)
		print('schedule mai: ',data)
		
	except Exception as e:
		message='error lagi: '+ str(e)
		print(message)
		return jsonify(message=message),400

	if data is not None:
		"""dari firestore"""
		sched,latency=read_from_firestore(data)
		print("Latency ngambil jadwal mai: ",latency)
		# print(sched)
		"""hardcode"""
		# sched=generate_schedule()
		tm=time.time()
		sched=sorting_sched(sched)
		sched_key=list(sched.keys())
		for i in sched_key:
			sched[i]['duration']=sched[i]['EndTime']-sched[i]['StartTime']
		print("latensi setelah ngambil dan sorting data: ",time.time()-tm)
		# print("schedule selain mai: ",sched)
		for i in sched_key:
			if(re.search("Machine",sched[i]['Id'])):
				if((sched[i]['EndTime']-sched[i]['StartTime'])>=data['duration']):
					"""Update details"""
					# # return
					# sched_ref.document(sched[i]['Id']).update({
					# 	'Description':data['Description']
					# })
					# print("banggggg")
					return jsonify("No schedule added"),200
				else:
					kkk=time.time()
					"""Hapus jadwal lama"""
					sched.pop(i)
					data['Id']=i
					processing_schedule_mai(percentages,data,sched)
					data.pop('duration')
					for i in sched:
						print(i,sched[i]['StartTime'],sched[i]['EndTime'])	
						sched[i].pop('duration')
					
					print("waktu eksekusi processing fungsi jadwal: ",time.time()-kkk)
					print("jadwal akhir: ",sched)
					print("jadwal mai akhir: ",data)
					try:
						kkkk=time.time()
						print("Execution time: ",datetime.now()-temp_time)

						sched_ref.document(data['Id']).update(data)
						for i in sched:
							to_firestore('update',sched[i])

						print("Sending Latency: ",datetime.now()-kkkk)
						print("Total time: ",datetime.now()-temp_time)

						return jsonify("Succesfully sent"),200
							
					except Exception as e:
						print("error bang: ",e)
						return jsonify("Error sending to firestore",data=e),400
			
		kkk=time.time()
		processing_schedule_mai(percentages,data,sched)

		data.pop('duration')
		# print(data)
		# print('schedule akhir:')
		for i in sched:	
			sched[i].pop('duration')
		
		print("waktu eksekusi processing fungsi jadwal: ",time.time()-kkk)
		print("jadwal akhir: ",sched)
		print("jadwal mai akhir: ",data)
		
		try:
			kkkk=time.time()
			to_firestore('send',data)
			print("Execution time: ",datetime.now()-temp_time)
			for i in sched:
				to_firestore('update',sched[i])
			print("Sending Latency jadwal mai: ",time.time()-kkkk)
			print("Total time jadwal mai: ",datetime.now()-temp_time)

			return jsonify("Succesfully sent"),200
				
		except Exception as e:
			print("error: ",e)
			return jsonify("Error sending to firestore",data=e),400

	else:
		return jsonify("No data received"),400

if __name__=="__main__":
	app.run(debug=True)