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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

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
# desc_mai_ref=db.collection('maintenance_description')
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


"""Triggerable processing"""
@app.route("/Random_forest",methods=["GET","POST"])
def Random_forest():
	try:
		ppp=time.time()
		inputer=request.json
		print('inputer: ',inputer)
		data=[[float(j) for j in inputer['data'].split(',')]]
		machine_id=inputer['machine_id']
		
		print("Ngambil data parameter ML: ",time.time()-ppp)

		# ppp=time.time()
		# print("bangg")
		# blob = bucket.blob(blob_name="{}.pkl".format(machine_id))
		# print("bangg")
		# model_file=io.BytesIO()
		# blob.download_to_file(model_file)
		# print("bangg")
		try:
			clf_RF = pickle.load(open("model/{}.pkl".format(machine_id),'rb'))
		except Exception as e:
			if(inputer['available']==0):
				try:
					blob = bucket.blob("{}.pkl".format(machine_id))
					db.collection('processing_params').document(machine_id).update({'available':1})
					blob.download_to_filename("model/{}.pkl".format(machine_id))
					clf_RF = pickle.load(open("model/{}.pkl".format(machine_id),'rb'))
				except Exception as e:
					# print("File not found. Error: {}".format(e))
					return jsonify("File not found. Error: {}".format(e)),400
			else:
				pred=np.array([1,0,0,0])
				# print("lagi download bang")

		try:
			pred=clf_RF.predict_proba(data)
		except:
			pass

		pred=pred.flatten().tolist()
		sorted_list = sorted(pred, reverse=True)
		first = pred.index(sorted_list[0])
		second = pred.index(sorted_list[1])
		third = pred.index(sorted_list[2])
		message_1,message_2,message_3=enterpret(first),enterpret(second),enterpret(third)
		result={
			'sorted_list':sorted_list,
			'1':[first,message_1],
			'2':[second,message_2],
			'3':[third,message_3]
		}
		
		return result,200
	
	except Exception as e:
		return jsonify("Error: {}".format(e)),400

def enterpret(data):
	if(data==0):
		return 'Normal'
	elif(data==1):
		return "Imbalance"
	elif(data==2):
		return "Horizontal Misalignment"
	elif(data==3):
		return 'Vertical Misalignment'

if(__name__=='__main__'):
	app.run(host="0.0.0.0",debug=True)