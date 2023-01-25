import serial

ser = serial.Serial('COM6', 9600)  # open serial port

while True:
    user_input = input("Enter '1' to turn on LED or '0' to off: ")
    if user_input == '1':
        ser.write(b'1')  # send '1' over serial to turn on LED
    elif user_input == '0':
        ser.write(b'0')
    else:
        print("Invalid input. Please enter '1' or '0'.")

ser.close()  # close serial port
