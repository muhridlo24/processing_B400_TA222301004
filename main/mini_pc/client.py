import socket
import firebase_admin
from firebase_admin import db, credentials, firestore
from datetime import datetime
import json
import pandas as pd
import numpy as np
import pywt
from scipy.signal import butter,lfilter
import csv
import sys
from multiprocessing import Pool
import re
import copy
import requests
import pickle
from sklearn.preprocessing import StandardScaler
from scipy import integrate
import psutil
import os


'''untuk mini PC'''
# cred_obj = credentials.Certificate('/home/operasional/Downloads/my-test-project-373716-firebase-adminsdk-nyuqz-bc57b26b33.json')

'''untuk di laptop'''
cred_obj = credentials.Certificate('D:/TA222301004/B400/mini_pc/my-test-project-373716-firebase-adminsdk-nyuqz-bc57b26b33.json')

default_app = firebase_admin.initialize_app(cred_obj)

db = firestore.client()

logs_ref = db.collection('Measurement_log')

initiation=1
 
# Sensor system
def sensor_output(x):
    temp, total_vib, dt, device_name = x.split(",")
    temp=float(temp)
    total_vib=float(total_vib)
    return temp, total_vib, dt, device_name

# Client msg
msgFromClient='Howdy Server'

#pre requisite 1 untuk processing
def read_file(nameFile,param):
	'''path untuk Laptop'''
	path = 'D:/TA222301004/B400/mini_pc/reference/reference.json'

	'''path untuk mini PC'''
	# path = "/home/operasional/Downloads/reference.json"

	try:
		with open(path,"r") as file:
			data = json.loads(file.read())

	except FileNotFoundError as i:
		print("File not found! {}",i)
  
	try:
		return data[nameFile][param]
	except KeyError:
		print("Value requested does not exist")

#pre requisite 2 untuk processing
def overwrite_file(nameFile,param,value):
	'''Path untuk mini PC'''
	# path = "/home/operasional/Downloads/reference.json"

	'''Path untuk Laptop'''
	path = "D:/TA222301004/B400/mini_pc/reference/reference.json"

	with open(path,"r") as file:
		dat = json.loads(file.read())
		print(dat)
		dat[nameFile][param] = value
  
	with open(path,"w") as file1:
		json.dump(dat,file1,indent=4)
		file1.close()

#processing temperature
def temperature_processing(deviceName,data):
	temperature_data = read_file(deviceName,"Temperature")
	key = list(temperature_data.keys())
	value = list(temperature_data.values())

	for i in range(len(value)):
		if(data>value[i]):
			return key[i]

	return "Normal"

#processing RPM
def rpm_processing(deviceName,data):
	rpm_data = read_file(deviceName,"RPM")
	key = list(rpm_data.keys())
	value = list(rpm_data.values())

	if (data>value[0]):
		return "Higher"
	elif (data<value[1]):
		return "Lower"
	else:
		return "Normal"

#get time delta
def get_timedelta(deviceName):
	timedelta = read_file(deviceName,"timedelta")
	return timedelta
	
#training processing RPM
def train_rpm(deviceName,data):
	old_rpm = read_file(deviceName,"Current RPM")
	if (old_rpm!=data):
		max_rpm = data*0.95
		min_rpm = data*1.05

	overwrite_file(deviceName,"RPM",{"Max":max_rpm,
                                       "Min":min_rpm})
  
	overwrite_file(deviceName,"Current RPM", data)

#Preprocessing Vibration
def vibration_preprocessing(val):
	if(len(vib_data)==1000):
		wavelet_new(data)
		vib_data=[]

	else:
		vib_data.append(val)
		print("Data length is not 1000!")

#Bandpass Filter (not applicable karena perlu filter baru)
def bandpass(x):
	# Cut-off frequency of the filter (in Hz)
	lowcut = 60
	highcut = 200

	# Sampling rate of the data (in Hz)
	fs = 1000

	# Filter order
	order = 2

	# Design the band-pass Butterworth filter
	nyquist = 0.5 * fs
	low = lowcut / nyquist
	high = highcut / nyquist
	b, a = butter(order, [low, high], btype='band', analog=False)

	# Apply the filter to the data
	y = lfilter(b, a, x)

	return y

#Fast Fourier Transform
def apply_fft(x, fs, num_samples):
    f = np.linspace(0.0, (fs/2.0), num_samples//2)
    freq_values = np.fft.fft(x)
    freq_values = 2.0/num_samples * np.abs(freq_values[0:num_samples//2])
    return f, freq_values

#Wavelet Transform
def wavelet_new(sinyal):
	temp = pywt.WaveletPacket(data=sinyal, wavelet='db4', mode='symmetric')
	x = [node.path for node in temp.get_level(3, 'natural')]
	temp_feature = [np.sum((temp[i].data)**2) for i in x]

	for i in x:
		new_wp = pywt.WaveletPacket(data = None, wavelet = 'db4', mode='symmetric',maxlevel=3)
		new_wp[i] = temp[i].data
		reconstructed_signal = new_wp.reconstruct(update = False) # Signal reconstruction from wavelet packet coefficients
		f, c = apply_fft(reconstructed_signal, 48000, len(reconstructed_signal))

		z = abs(c)

		# Find  m  highest amplitudes of the spectrum and their corresponding frequencies:
		maximal_idx = np.argpartition(z, -1)[-1:]
		high_amp = z[maximal_idx]
		high_freq = f[maximal_idx]
		feature = high_amp*high_freq
		temp_feature.append(list(high_amp)[0])
		temp_feature.append(list(high_freq)[0])
		temp_feature.append(list(feature)[0])

	return temp_feature

#Send Data to Firestore
def send_firestore(time,line,plant,zona,deviceName,temp,rpm,vib_mean):
	log={
	str(deviceName):{
		"line":line,
		"performance_log":{
			"rpm":rpm,
			"temperature":temp,
			"vibration":vib_mean
		},
		"plant":plant,
		"timestamp":time,
		"zona":zona
	}
	}
	logs_ref.document(time).set(log, merge=True)

# #Send Processed Data to Firestore
# def send_processed(time,line,plant,zona,deviceName,temp,rpm,vib_mean):
# 	log={
# 	str(deviceName):{
# 		"line":line,
# 		"processed":{
# 		"rpm":rpm,
# 		"temperature":temp,
# 		"vibration":vib_mean
# 		},
# 		"plant":plant,
# 		"zona":zona
# 	}
# 	}
# 	processed_ref.document(time).set(log, merge=True)

#Save data to csv format
def to_excel(filename,new_data):
	with open(filename, 'a', newline='') as file:
		writer = csv.writer(file)
		for row in new_data:
			writer.writerow(new_data)

#Trend Calculation for Temperature and RPM Data
def trend_calc(data):
	mean = np.mean(data)
	std = np.std(data)
	deviations = [x - mean for x in data]
	squared_deviations = [x ** 2 for x in deviations]
	sum_of_squared_deviations = sum(squared_deviations)
	slope = sum_of_squared_deviations / (len(data) - 1)
	return [mean,std,slope]

def accel_to_rmsvelo(data):
	data = (np.array(data)*9.81*1000)
	data = data-np.mean(data)
	integrated = integrate.cumtrapz(data, dx = 1.0/1000, initial = 0)

	rms = np.sqrt(np.mean(np.array(integrated)**2))
	# print("Data: {}   Kecepatan: {}   RMS: {}".format(data,integrated,rms))
	return rms

#Stability of Data
def stability(deviceName,rpm_data):
	mean_rpm = sum(rpm_data) / len(rpm_data)

	std_dev_rpm = np.std(rpm_data)
	cv = std_dev_rpm / mean_rpm

	if cv < read_file(deviceName,'CV'):
		stability = 'Stable'
	else:
		stability = 'Unstable'

	return [mean_rpm,std_dev_rpm,stability]

def send_to_cloud_run(time,line,plant,zona,deviceName,temp,temp_stat,
	rpm,rpm_stat,rpm_stability,rpm_std,vib_processed):
	log={
	"time":time,
	"machine_id":deviceName,
	"line":line,
	"rpm":{
		"stability":rpm_stability,
		"stat":rpm_stat,
		"std":rpm_std,
		"value":rpm
	},
	"temperature":{
		"status":temp_stat,
		"value":temp
	},
	"vibration":",".join(map(str,vib_processed)),
	"plant":plant,
	"zona":zona
	}
	print(log)
	return log

def rms_detect(deviceName,data):
	rpm_data = read_file(deviceName,"Vibration")
	key = list(rpm_data.keys())
	value = list(rpm_data.values())

	for i in range(len(value)):
		if(data>=value[i]):
			return key[i]

	return "Good"

def preprocessing(total_data):
	temp_time=datetime.now()
	print("Starting {} at: {}".format(total_data['machine_id'],temp_time))
	print("Total Size: ",sys.getsizeof(total_data))
	
	"""---------------------------------"""
	result={}

	for key,value in total_data['performance_log'].items():
		if(re.match("rpm",key)):
			rpm_mean,rpm_std,rpm_stability=stability(total_data['machine_id'],total_data['performance_log'][key])
			rpm_processed=rpm_processing(total_data['machine_id'],float(rpm_mean))
			rpm_processed_1=rpm_processing(total_data['machine_id'],max(total_data['performance_log'][key]))
			rpm_processed_2=rpm_processing(total_data['machine_id'],min(total_data['performance_log'][key]))
		
			result[key]={
				"stability":rpm_stability,
				"stat":[rpm_processed,rpm_processed_1,rpm_processed_2],
				"std":rpm_std,
				"value":[rpm_mean,max(total_data['performance_log'][key]),min(total_data['performance_log'][key])]
			}
			print("{} selesai dengan waktu eksekusi: {}".format(key,datetime.now()-temp_time))

		elif(re.match("vibration",key)):
			total_data['performance_log'][key] = np.array(total_data['performance_log'][key]).flatten().tolist()
			vib_processed=wavelet_new(total_data['performance_log'][key])
			sc = pickle.load(open('D:/TA222301004/B400/mini_pc/reference/scaler.pkl','rb'))

			wavelet=",".join(map(str,sc.transform(np.array(vib_processed).reshape(1,-1)).flatten().tolist()))
			rms_value=accel_to_rmsvelo(np.array(total_data['performance_log'][key]).flatten().tolist())
			rms_category=rms_detect(total_data['machine_id'],rms_value)

			result[key]={
				"wavelet":wavelet,
				"rms_value":rms_value,
				"rms_category":rms_category
			}

			print("{} selesai dengan waktu eksekusi: {}".format(key,datetime.now()-temp_time))

		elif(re.match("temperature",key)):
			temp_mean=np.mean(total_data['performance_log']['temperature'])
			trend_temp=trend_calc(total_data['performance_log']['temperature'])
			temp_processed=temperature_processing(total_data['machine_id'],temp_mean)
			temp_processed_1=temperature_processing(total_data['machine_id'],max(total_data['performance_log'][key]))
			temp_processed_2=temperature_processing(total_data['machine_id'],min(total_data['performance_log'][key]))

			result[key]={
				"status":[temp_processed,temp_processed_1,temp_processed_2],
				"mean":temp_mean,
				"trend":trend_temp[2],
				"value":[temp_mean,max(total_data['performance_log'][key]),min(total_data['performance_log'][key])]
			}

			print("{} selesai dengan waktu eksekusi: {}".format(key,datetime.now()-temp_time))

	result['time']=datetime.now().isoformat("@","minutes")
	result['machine_id']=total_data['machine_id']
	result['line']='LHL'
	result['plant']="J2"
	result['zona']="Zona A"

	# to_be_saved=[timestamp,device_name,line,plant,zona,temp_mean,
	# 	trend_temp,rpm_mean,rpm_std,stability_rpm,total_vib]
	# to_excel("processed_data.csv",to_be_saved)
	print(result)
	# logs_ref.document(data[list(data.keys())[0]]['timestamp']).set(data, merge=True)
	# resp = requests.post("https://processing-k4ulq4ld5a-et.a.run.app/warning-machine",json=result)
	
	resp = requests.post("http://127.0.0.1:5000/warning-machine",json=result)

	res_time=datetime.now()-temp_time
	# print(result)
	print("Ended {} at: {}".format(total_data['machine_id'],res_time))
	return res_time

if __name__=='__main__':
	rpm_total=[]
	temp_total=[]
	vibration_data=[]
	time_reference=datetime.now()
	time_reference_3=datetime.now()
	data=""
	total_data={}
	enable=0
	temp={}
	receive=0
	print('coba test')
	while 1:
		'''untuk testing tanpa raspi'''
		if((datetime.now()-time_reference_3).seconds>=1):
			data={
				"LHL01":{
					"timestamp":datetime.now().isoformat("@","seconds"),
					"performance_log":{
						"temperature":round(np.random.uniform(80,90),3),
						"rpm":round(np.random.uniform(2500,2600),3),
						"vibration_1":','.join(map(str,np.random.uniform(-0.01,0.01,1000))),
						"vibration_2":','.join(map(str,np.random.uniform(-0.01,0.01,1000)))
					}
				}
			}
			line="LHL"
			zona="Zona A"
			plant="J2"
			time_reference_3=datetime.now()
			data=json.dumps(data)
			receive=1

		'''untuk Testing dengan raspi'''
		# Encode data to server
		# while(receive==0):
		# 	bytesToSend=msgFromClient.encode('utf-8')
		# 	serverAddress=("192.168.43.96", 3333)
		# 	bufferSize=1024

		# 	# Send byte data
		# 	UDPClient=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		# 	UDPClient.sendto(bytesToSend, serverAddress)

		# 	# Recv 
		# 	data,address=UDPClient.recvfrom(bufferSize)
		# 	data=data.decode('utf-8')
		# 	print(data)
		# 	print("halo bang")
		# 	receive=1

		if(receive==1):
			temp_time=datetime.now()
			data=json.loads(data)
			data[list(data.keys())[0]]['line']="LHL"
			data[list(data.keys())[0]]['zona']="Zona A"
			data[list(data.keys())[0]]['plant']="J2"

			# temp, total_vib, timestamp, device_name = sensor_output(data)
			
			print('---------------------------------')
			#print('Server IP Address: ', address[0])
			#print('Server Port: ', address[1])
			print('Data from Server')
			print("Machine: ",list(data.keys())[0])
			print("Temperature: {}C".format(data[list(data.keys())[0]]['performance_log']['temperature']))
			# print("Acceleration: {}m/s^2".format(data[list(data.keys())[0]]['performance_log']['vibration']))
			print("RPM: {}".format(data[list(data.keys())[0]]['performance_log']['rpm']))
		
			for key,value in (data[list(data.keys())[0]]['performance_log'].items()):
				if(re.match("vibration",key)):
					data[list(data.keys())[0]]['performance_log'][key] = [float(i) for i in value.split(',')]

			if(list(data.keys())[0] not in list(total_data.keys())):
				total_data[list(data.keys())[0]]=copy.deepcopy(data[list(data.keys())[0]])
				total_data[list(data.keys())[0]]['machine_id']=list(data.keys())[0]
				total_data[list(data.keys())[0]].pop('timestamp')

				for i,value in (total_data[list(data.keys())[0]]['performance_log'].items()):
					total_data[list(data.keys())[0]]['performance_log'][i]=[value]

					if(re.match('vibration',i)):
						data[list(data.keys())[0]]['performance_log'][i]= accel_to_rmsvelo(data[list(data.keys())[0]]['performance_log'][i])

				total_data[list(data.keys())[0]]['time_reference']=datetime.now()

			else:
				for key, value in data[list(data.keys())[0]]['performance_log'].items():
					if key in total_data[list(data.keys())[0]]['performance_log']:
						total_data[list(data.keys())[0]]['performance_log'][key].append(value)
					
					if(re.match('vibration',key)):
						# print(data[list(data.keys())[0]]['performance_log'][key])
						print("RMS Velocity: ",accel_to_rmsvelo(data[list(data.keys())[0]]['performance_log'][key]))
						data[list(data.keys())[0]]['performance_log'][key]= accel_to_rmsvelo(data[list(data.keys())[0]]['performance_log'][key])

			
			total_data[list(data.keys())[0]]['timedelta']=(datetime.now()-total_data[list(data.keys())[0]]['time_reference']).seconds
			
			"""Save To Local File"""
			# to_be_saved=[timestamp,device_name,line,plant,zona,temp,rpm,total_vib]
			# to_excel("data.csv",to_be_saved)
			print("Time delta: ",total_data[list(total_data.keys())[0]]['timedelta'])

			print("Data: ",data)
			print(total_data['LHL01']['time_reference'])
			# print(data[list(data.keys())[0]]['timestamp'],data)

			# logs_ref.document(data[list(data.keys())[0]]['timestamp']).set(data, merge=True)

			print("execution time: ",datetime.now()-temp_time)
			print(datetime.now()-time_reference)
			# to_be_saved=[]
			key_machine=list(total_data.keys())
			key_processing=[]

			for i in (key_machine):
				if(total_data[i]['timedelta']>get_timedelta(deviceName=i)):
					enable=1
					temp_timestamp=datetime.now()
					key_processing.append(i)

			if(enable==1):
				print("Processing started")
				# with Pool(processes=len(key_processing)) as pool:
				print("test doang ini bang: ",datetime.now())
				for i in map(preprocessing,[total_data[j] for j in key_processing]):
					print("Done processed with latency of: ",i)
				
				for i in key_processing:
					total_data.pop(i)
				
				key_processing=[]
				enable=0
				time_reference=datetime.now()

			data=""
			receive=0
		# mendapatkan pid (process id) dari program python saat ini
		# 	pid = os.getpid()
		# 	py = psutil.Process(pid)

		# 	# menampilkan penggunaan memori dan CPU saat ini
		# 	memory_use = py.memory_info()[0] / 2.**30  # memori dalam gigabytes
		# 	cpu_use = py.cpu_percent()

		# 	print(f"Memory use: {memory_use} GB")
		# 	print(f"CPU use: {cpu_use}%")
		# 	print(f"CPU Usage Percentage: ",psutil.disk_usage('/'))
		# # elif(((datetime.now()-time_reference).minute==30) and data==""):
		# # 	#mode sleep
		# # 	pass