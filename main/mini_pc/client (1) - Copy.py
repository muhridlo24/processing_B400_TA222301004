import socket
import time
import threading
import random
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

'''untuk mini PC'''
# cred_obj = credentials.Certificate('/home/operasional/Downloads/my-test-project-373716-firebase-adminsdk-nyuqz-bc57b26b33.json')

'''untuk di laptop'''
cred_obj = credentials.Certificate('mini_pc/my-test-project-373716-firebase-adminsdk-nyuqz-bc57b26b33.json')

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
	path = 'mini_pc/reference.json'

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
	path = "/home/operasional/Downloads/reference.json"

	'''Path untuk Laptop'''
	path = "mini_pc/reference.json"

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
			print(data,value[i])
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

#Stability of Data
def stability(rpm_data):
	mean_rpm = sum(rpm_data) / len(rpm_data)

	std_dev_rpm = np.std(rpm_data)
	cv = std_dev_rpm / mean_rpm

	if cv < 0.01:
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

if __name__=='__main__':
	rpm_total=[]
	temp_total=[]
	vibration_data=[]
	time_reference=datetime.now()
	time_reference_2=datetime.now()
	time_reference_3=datetime.now()
	data=""
	total_data={}
	while 1:
		# '''untuk testing tanpa raspi'''
		# if((datetime.now()-time_reference_3).microseconds>=800000):
		# 	# Encode data to server
		# 	bytesToSend=msgFromClient.encode('utf-8')
		# 	serverAddress=("192.168.43.96", 3333)
		# 	bufferSize=1024

		# 	# Send byte data
		# 	UDPClient=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		# 	UDPClient.sendto(bytesToSend, serverAddress)

		# 	# Recv 
		# 	#data,address=UDPClient.recvfrom(bufferSize)
		# 	#data=data.decode('utf-8')
		# 	data=",".join(map(str,[np.random.uniform(60,80),np.random.uniform(0,1),
		# 		datetime.now().isoformat("@","seconds"),"LHL01"]))
		
		# 	print(data)
		# 	line="LHL"
		# 	zona="Zona A"
		# 	plant="J2"
		# 	time_reference_3=datetime.now()

		# if((datetime.now()-time_reference_2).microseconds>=500000):
		# 	# Encode data to server
		# 	bytesToSend=msgFromClient.encode('utf-8')
		# 	serverAddress=("192.168.43.96", 3333)
		# 	bufferSize=1024

		# 	# Send byte data
		# 	UDPClient=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		# 	UDPClient.sendto(bytesToSend, serverAddress)

		# 	# Recv 
		# 	# data,address=UDPClient.recvfrom(bufferSize)
		# 	# data=data.decode('utf-8')
		# 	data=",".join(map(str,[np.random.uniform(60,80),np.random.uniform(0,1),
		# 		datetime.now().isoformat("@","seconds"),"LHL02"]))
		
		# 	print(data)
		# 	line="LHL"
		# 	zona="Zona A"
		# 	plant="J2"
		# 	time_reference_2=datetime.now()
		
		'''untuk Testing dengan raspi'''
		# Encode data to server
		bytesToSend=msgFromClient.encode('utf-8')
		serverAddress=("192.168.43.96", 3333)
		bufferSize=1024

		# Send byte data
		UDPClient=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		UDPClient.sendto(bytesToSend, serverAddress)

		# Recv 
		data,address=UDPClient.recvfrom(bufferSize)
		data=data.decode('utf-8')	
		line="LHL"
		zona="Zona A"
		plant="J2"

		if(data!=""):
			temp_time=datetime.now()
			data=json.loads(data)
			# temp, total_vib, timestamp, device_name = sensor_output(data)

			# Random rpm
			rpm=random.randint(2800,2900)
			
			print('---------------------------------')
			#print('Server IP Address: ', address[0])
			#print('Server Port: ', address[1])
			print('Data from Server')
			print("Machine: ",data['device_name'])
			print("Temperature: {}C".format(data['temperature']))
			print("Acceleration: {}m/s^2".format(data['vibration']))
			print("RPM: {}".format(rpm))
			
			if(data['device_name'] not in total_data.keys()):
				total_data[data['device_name']]={
					'time_reference':datetime.now(),
					'timedelta':1,
					'RPM':[],
					'Vibration':[],
					'Temperature':[]
				}

			if(rpm!=""):
				total_data[data['device_name']]['RPM'].append(float(rpm))
				
			if(data['temperature']!=""):
				total_data[data['device_name']]['Temperature'].append(float(temp))
				
			if(data['vibration']!=""):
				total_data[data['device_name']]['Vibration'].append(float(total_vib))
			
			total_data[data['device_name']]['timedelta']=(datetime.now()-total_data[data['device_name']]['time_reference']).seconds
					
			"""Save To Local File"""
			# to_be_saved=[timestamp,device_name,line,plant,zona,temp,rpm,total_vib]
			# to_excel("data.csv",to_be_saved)

			# data = send_firestore(timestamp,line,plant,zona,device_name,
			# 	temp,rpm,total_vib)

			print("execution time: ",datetime.now()-temp_time)
			print(datetime.now()-time_reference)
			# to_be_saved=[]
			key_machine=list(total_data.keys())
			for i in (key_machine):
				if(total_data[i]['timedelta']>58):
					print("Total Size: ",sys.getsizeof(total_data[i]))
					rpm_mean,rpm_std,stability_rpm=stability(total_data[i]['RPM'])
					temp_mean=np.mean(total_data[i]['Temperature'])
					trend_temp=trend_calc(total_data[i]['Temperature'])
					temp_processed=temperature_processing(data['device_name'],float(temp_mean))
					rpm_processed=rpm_processing(data['device_name'],float(rpm_mean))
					vib_processed=wavelet_new(total_data[i]['Vibration'])
					
					"""---------------------------------"""
					# Data from RPI

					rpm_total=[]
					temp_total=[]
					vibration_data=[]
					time_reference=datetime.now()
					
					# to_be_saved=[timestamp,device_name,line,plant,zona,temp_mean,
					# 	trend_temp,rpm_mean,rpm_std,stability_rpm,total_vib]
					# to_excel("processed_data.csv",to_be_saved)
					data = send_firestore(data['timestamp'],line,plant,zona,data['device_name'],
						data['temperature'],rpm,np.mean(data['vibration']))
					sent=send_to_cloud_run(time,line,plant,zona,data['device_name'],temp_mean,
						temp_processed,rpm_mean,rpm_processed,stability_rpm,rpm_std,vib_processed)
					
					total_data.pop(i)
					#resp = requests.post("http://localhost:5000/warning-machine",json=sent)

			data=""


			



