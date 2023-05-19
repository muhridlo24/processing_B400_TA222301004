import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import pywt
import numpy as np
from scipy.signal import butter, lfilter
from datetime import datetime
import json, requests
import csv
import random
import sys

cred_obj = credentials.Certificate('D:/b400/mini_pc/my-test-project-373716-firebase-adminsdk-nyuqz-bc57b26b33.json')
default_app = firebase_admin.initialize_app(cred_obj)

db = firestore.client()

logs_ref = db.collection('logs')
processed_ref = db.collection('performance')
warning_ref = db.collection('warning')

#pre requisite 1 untuk processing
def read_file(nameFile,param):
  path = "D:/b400/mini_pc/reference.json"

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
  path = "D:/b400/mini_pc/reference.json"
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
def vibration_preprocessing(vib_data,val):
  if(len(vib_data)>=1000):
    wavelet_new(vib_data)
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

def wav_feature_new(dat_1,length,col):
  merged_dat = pd.DataFrame()

  for i in range(len(dat_1.fault.unique())):
    temp = dat_1.loc[dat_1.fault==dat_1.fault.unique()[i]]
    wavelett = [wavelet_new(temp.iloc[xx:xx+length,col].values)
                if (xx+length<temp.shape[0]) else wavelet_new(temp.iloc[xx:temp.shape[0],col].values)
                for xx in range(0,temp.shape[0],length)]

    fault = [temp.iloc[0,-1] for k in range(0,temp.shape[0],length)]
      
    res_dat = pd.concat([pd.DataFrame(wavelett),
                        pd.DataFrame(fault)],axis=1)
    
    merged_dat = pd.concat([merged_dat,res_dat],axis=0)
    
  return merged_dat

#Send data to Firebase Realtime Database
# def send_realtime(time,line,plant,zona,deviceName,temp,rpm,vib_mean):
#   ref_1 = ref.child(str(deviceName))
#   # ref_1.child('id').push().set({"Timestamp":time,
#   #                         "Value":deviceName})
#   # ref_1.child('line').push().set({"Timestamp":time,
#   #                                 "Value":line})
#   # ref_1.child('performance_log').push().set({"Timestamp": time,
#   #                                "Temperature": temp,
#   #                                "RPM": rpm,
#   #                                "Vibration": vib_mean})
#   # ref_1.child('plant').push().set({"Timestamp":time,
#   #                                  "Value":plant})
#   # ref_1.child('zona').push().set({"Timestamp":time,
#   #                                 "Value":zona})
#   ref_1.push().set({"Timestamp":time,
#                     "id":deviceName,
#                     "line":line,
#                     "performance_log":{"Temperature":temp,
#                                         "RPM":rpm,
#                                         "Vibration":vib_mean},
#                     "plant":plant,
#                     "zona":zona})

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
      "zona":zona
    }
  }
  logs_ref.document(time).set(log, merge=True)

#Send Processed Data to Firestore
def send_processed(time,line,plant,zona,deviceName,temp,rpm,vib_mean):
  log={
    str(deviceName):{
      "line":line,
      "processed":{
        "rpm":rpm,
        "temperature":temp,
        "vibration":vib_mean
        },
      "plant":plant,
      "zona":zona
    }
  }
  processed_ref.document(time).set(log, merge=True)

#Save data to csv format
def to_excel(new_data):
  with open('data.csv', 'a', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    writer.writerow(new_data)

#Trend Calculation for Temperature and RPM Data
def trend_calc(data):
  mean = np.mean(data)
  std = np.std(data)
  deviations = [x - mean for x in data]
  squared_deviations = [x ** 2 for x in deviations]
  sum_of_squared_deviations = sum(squared_deviations)
  slope = sum_of_squared_deviations / (len(data) - 1)
  return slope

#Stability of Data
def stability(rpm_data):
  mean_rpm = sum(rpm_data) / len(rpm_data)

  std_dev_rpm = np.std(rpm_data)
  cv = std_dev_rpm / mean_rpm

  if cv < 0.01:
      stability = 'Stable'
  else:
      stability = 'Unstable'

  return mean_rpm,std_dev_rpm,stability

#send to cloud run
def send_to_cloud_run(time,line,plant,zona,deviceName,temp,temp_stat,rpm,rpm_stat,rpm_stability,rpm_std,vib_processed):
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
  return log

#Flow
def main():
  #Terima data
  tm = datetime.now()
  deviceName="LHL01"
  line="Line 1"
  plant="J2"
  zona="Zona A"
  time=datetime.now().isoformat("@", "minutes")

  #Split data temperatur, RPM, dan vibrasi
  rpm=[random.uniform(2450,2550) for i in range(60)]
  temp=[random.uniform(50,80) for i in range(60)]
  vib_mean=0.5

  vibration=pd.read_csv('D:/b400/mini_pc/2003.10.22.12.06.24',sep='\t',header=None).iloc[:1000,0].values

  #processing temperatur
  print(datetime.now()-tm)
  temp_mean=np.mean(temp)
  trend_temp=trend_calc(temp)
  processed_temp=temperature_processing(deviceName,temp_mean)

  print(datetime.now()-tm)
  rpm_mean,rpm_std,stability_rpm=stability(rpm)
  processed_rpm=rpm_processing(deviceName,rpm_mean)

  print(datetime.now()-tm)

  #preprocessing data vibrasi
  vib_processed=wavelet_new(vibration)
  print(datetime.now()-tm)

  #Panggil cloud Run untuk processing vibration
  sent=send_to_cloud_run(time,line,plant,zona,deviceName,
                temp_mean,processed_temp,rpm_mean,processed_rpm,stability_rpm,rpm_std,vib_processed)
  
  print(datetime.now()-tm)
  resp = requests.post("http://localhost:5000/warning-machine",json=sent)

  print(resp)
  print(datetime.now()-tm)

  #Kirim data temperatur, RPM, dan vibrasi ke firestore
  send_firestore(time,line,plant,zona,deviceName,
                 temp_mean,rpm_mean,vib_mean)

  print(datetime.now()-tm)
  #Kirim data hasil processing temperatur dan RPM
  send_processed(time,line,plant,zona,deviceName,
                processed_temp,processed_rpm,vib_mean)

  print(datetime.now()-tm)

  to_excel([time,deviceName,line,plant,zona,temp_mean,rpm_mean,vib_mean])
  print(datetime.now()-tm)

main()