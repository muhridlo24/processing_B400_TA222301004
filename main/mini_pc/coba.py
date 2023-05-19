import time
import random

# fungsi untuk membaca data sensor temperatur
def read_temperature():
    # kode untuk membaca sensor temperatur disini
    # contoh data dummy
    temperature = round(random.uniform(20.0, 30.0), 2) #ini diganti sama bacaan sensor temperatur
    return temperature

# fungsi untuk membaca data sensor vibrasi
def read_vibration():
    # kode untuk membaca sensor vibrasi disini
    # contoh data dummy
    vibration_data = []
    temp=time.time()
    temp_2=time.time()
    while(time.time()-temp<1):
        if((time.time()-temp_2)>=0.0005):
            vibration_data.append(round(random.uniform(0.0, 1.0), 2)) #round(random.uniform(0.0, 1.0), 2) diganti sama bacaan sensor vibrasi
            temp_2=time.time()

    return vibration_data

# fungsi untuk mengumpulkan data dari kedua sensor ke dalam dictionary
def collect_data():
    temperature = read_temperature()
    vibration = read_vibration()

    """di bawah ini ganti sama struktur data yg gw kirim"""
    data_dict = {'temperature': temperature, 'vibration': vibration}
    return data_dict

# program utama
while True:
    start_time = time.time() # waktu awal
    data = collect_data()

    elapsed_time = time.time() - start_time # waktu yang dibutuhkan untuk membaca sensor

    """ini diganti sama ngirim data ke mini PC"""
    print("Execution time: ",elapsed_time)
    time.sleep(max(0, 1 - elapsed_time)) # delay agar waktu bacaan sensor tepat 1 detik