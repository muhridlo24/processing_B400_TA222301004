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


"""Triggerable processing"""
@app.route("/rnn-janet",methods=["POST"])
def RNN_Janet():
	try:
		bg=time.time()
		ppp=time.time()
		inputer=request.json
		data=[[float(j) for j in inputer['data'].split(',')]]
		machine_id=inputer['machine_id']
		TIME_STEPS = inputer['TIME_STEPS']
		NUM_UNITS = inputer['NUM_UNITS']
		LEARNING_RATE = inputer['LEARNING_RATE']
		NUM_OUTPUTS=inputer['NUM_OUTPUTS']
		message=""
		title=[]

		data = np.array(data).reshape(1, -1, 1)
		model = JANet(inputs=1, cells=NUM_UNITS, num_outputs=NUM_OUTPUTS, num_timesteps=TIME_STEPS,output_activation=nn.Softmax(dim=1))
		print("ngambil data params: ",time.time()-ppp)
		try:
			ppp=time.time()
			blob = bucket.blob(blob_name="{}.pkl".format(machine_id))
			data_model=blob.download_as_bytes()
			try:
				torched=torch.load(io.BytesIO(data_model))
				model.load_state_dict(torched,strict=False)
				optimizer = optim.Adam(list(model.parameters()), lr=LEARNING_RATE)
				optimizer.zero_grad()
				print("waktu eksekusi ngambil model: ",time.time()-ppp)
				try:
					ppp=time.time()
					pred=model(torch.from_numpy(data.astype('float32')).to(device))
					pred=pred.cpu().detach().numpy().flatten().tolist()
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
					print("prediksi: ",time.time()-ppp)
					print("execution time total: ",time.time()-bg)
					return result
				
				except Exception as e:
					message="Error ketika melakukan prediksi. Error: {}".format(e)
					return jsonify(message),400
				
			except Exception as e:
				message="Failed to load pickle file to model. Error: {}".format(e)
				return jsonify(message),400

		except Exception as e:
			message="File model tidak ditemukan."
			return jsonify(message),400
	
	except Exception as e:
		print("errorrrr: ",e)
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
	elif(data==4):
		return 'others'

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


if(__name__=="__main__"):
	app.run(host="0.0.0.0",debug=True)