import serial
import tm1637
import time

ser = serial.Serial(
    port ='/dev/serial0',
    baudrate = 9600,
    timeout = 1
)
tm = tm1637.TM1637(clk=23,dio=24)

def send_data(data):
    if ser.is_open:
        ser.write(data.encode('utf-8'))
        print(f"已傳送:{data}")

def receive_data():
    if ser.is_open:
        incoming = ser.readline().decode('utf-8').strip()
        if incoming:            
            print(f"收到資料:{incoming}")
            try:
                numeric_val = int(incoming)
                tm.number(numeric_val)
            except ValueError:
                print(f"警告: 收到的資料 '{incoming}' 不是純數字，無法顯示")
        return incoming

try:
    while True:
        send_data("1234")
        time.sleep(1)
        receive_data()
        time.sleep(1)
except KeyboardInterrupt:
    print("中斷")

finally:
    if ser.is_open:
        ser.close()
        print("串列port已關閉")