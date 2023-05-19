import json
import time
import requests
 
# function to add to JSON
def write_json(new_data, filename='total_data.json'):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        # print(new_data.values())
        file_data[list(new_data.keys())[0]]=new_data[list(new_data.keys())[0]]
        file.seek(0)
        json.dump(file_data, file, indent = 4)
 
    # python object to be appended
y = {"BANGG":
     {
         "emp_name":"Nikhil",
         "email": "nikhil@geeksforgeeks.org",
         "job_profile": "Full Time"
    }}
     
# write_json(y)

# with open('cache/data.json','r+') as file:
#     reader=json.load(file)

#     print(reader[-1])
#     if(len(reader)>0):
#     #threading untuk kirim cache
#         for i in range(len(reader)):
#             print(0,reader[0])
#             resp = requests.post("https://processing-k4ulq4ld5a-et.a.run.app/warning-machine",json=reader[0])
#             print(resp)
#             print(resp.elapsed.total_seconds())
#             """If not then kirim threading"""
#             reader.remove(reader[0])

#     with open('cache/data.json','w') as file:
#         json.dump(reader, file, indent = 4)
#         file.close()


{
    'temperature': {
        'status': ['Danger', 'Danger', 'Danger'], 
        'mean': 93.9931, 
        'trend': 13.558397877777772, 
        'value': [93.9931, 99.977, 90.112]},
    'rpm': {
        'stability': 'Unstable', 
        'stat': ['Higher', 'Higher', 'Higher'], 
        'std': 27.164122548685448, 
        'value': [2479.6876, 2527.742, 2435.515]}, 
    'vibration': {
        'wavelet': '65.773324826184,-0.17186937723599396,-0.2321589336505512,6.710816134604,-0.4045044551903263,8.878253251907084,4.303038514299226,0.16127919042029737,-0.17279636630839826,487176.24368349224,200.39210921121276,-0.27164187315088245,87181.04081241025,85.0569949379523,-1.6577299499302964,159548.3718021134,91.15365776618292,-4.596156967357788,10013.515672690568,16.404459526722615,-12.269111297617936,4918.315653763948,146.3083973654684,-8.441902320930513,13188.034398867545,793.0707139860992,-26.550151264898684,23113.543077391587,416.2685365835738,-20.653102898167948,30449.72708073833,39.404591256361556,-15.850649158505933,-0.5133704704912682,12.131183953106207,-19.300127551115555,3.9654251581908446', 
        'rms_value': 0.197428793602048,
        'rms_category': 'Good'
        }, 
    'time': '2023-05-15@07:43', 
    'machine_id': 'LHL01', 
    'line': 'LHL', 
    'plant': 'J2', 
    'zona': 'Zona A'
}

import numpy as np

print(np.array([[1,2,3],[4,5,6]]).flatten().tolist())


import sys

my_variable = 'c'
size = sys.getsizeof(my_variable)

# print("Size of my_variable:", size, "bytes")
print(my_variable.__sizeof__())

arr=[]
from scipy import integrate
def accel_to_rmsvelo(data):
	data = (np.array(data)*9.81*1000)
	data = data-np.mean(data)
	integrated = integrate.cumtrapz(data, dx = 1.0/1000, initial = 0)

	rms = np.sqrt(np.mean(np.array(integrated)**2))
	return rms


with open('rekap.json','r+') as file:
    reader=json.load(file)
    temp=reader[list(reader.keys())[0]]['performance_log']['vibration']

    for i in range(len(temp)):
        arr=arr+temp[i]
    
    print(accel_to_rmsvelo(arr))


import pandas as pd
import openpyxl
df_json = pd.read_json('rekap.json')
df_json.to_excel('wavelet.xlsx')

print(len(arr))