import serial

ser = serial.Serial('COM6', 9600)  # open serial port


def on_uv_light():
    ser.write(b'1')  # send '1' over serial to turn on LED


def off_uv_light():
    ser.write(b'0')
