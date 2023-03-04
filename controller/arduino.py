import serial
import time

# Define the serial port and baud rate for the Arduino
ser = serial.Serial('COM6', 9600, timeout=1)

# Wait for the Arduino to initialize
time.sleep(2)


def uv_on():
    ser.write(b's')


def uv_off():
    ser.write(b'x')


# while True:
#     # Read the value from the serial port and print it to the console
#     ldrValue = ser.readline().decode('utf-8').rstrip()
#     print("LDR value: ", ldrValue)
#
#     # Check for keyboard input and send command to Arduino
#     command = input("Enter command: ")
#     if command == 's':
#         ser.write(b's')
#     elif command == 'x':
#         ser.write(b'x')

