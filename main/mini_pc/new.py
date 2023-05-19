import time
import sched

# fungsi untuk membaca sensor temperatur
def read_temp():
    # kode untuk membaca sensor temperatur
    # dan mengembalikan nilai temperatur dalam float
    return 25.5

# fungsi untuk membaca sensor vibrasi
def read_vibration():
    # kode untuk membaca sensor vibrasi
    # dan mengembalikan nilai vibrasi dalam float
    return 3.2

# inisialisasi scheduler
scheduler = sched.scheduler(time.time, time.sleep)

# fungsi untuk membaca sensor temperatur
def read_temp_wrapper():
    # membaca sensor temperatur
    temp = read_temp()
    # menjadwalkan pembacaan sensor temperatur selama 1 detik ke depan
    scheduler.enter(1, 1, read_temp_wrapper)
    # mengembalikan nilai temperatur dalam dictionary
    return {'temperature': temp}

# fungsi untuk membaca sensor vibrasi
def read_vibration_wrapper():
    # membaca sensor vibrasi sebanyak 1000 kali dengan interval 1 ms
    vibration = [read_vibration() for i in range(1000)]
    # menjadwalkan pembacaan sensor vibrasi selama 1 detik ke depan
    scheduler.enter(1, 2, read_vibration_wrapper)
    # mengembalikan nilai vibrasi dalam dictionary
    return {'vibration': vibration}

# menjalankan scheduler untuk pertama kali
scheduler.enter(0, 1, read_temp_wrapper)
scheduler.enter(0, 2, read_vibration_wrapper)

# menjalankan scheduler secara terus-menerus
scheduler.run()