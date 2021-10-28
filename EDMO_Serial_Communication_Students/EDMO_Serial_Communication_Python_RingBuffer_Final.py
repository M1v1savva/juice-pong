import serial 
import numpy as np
from timeit import default_timer as timer

######## INIT serial communication variables################################
ser = serial.Serial('COM18', 57600) #select com-port and the serial com baud rate
ser.flushInput() #empty the serial buffer
ser_input=[] #emtpy the serial input array
current_time = 0 
prev_time = 0
interval = 0.5 #serial read update time

#Example commands to drive the motors
commandString ='0,0,25,0,0,45,1000,0,0,5000\n' #data string to send to arduino
commandString1 ='1,2,5,0,2,10,1000,2,30,2000\n' #data string to send to arduino
commandString2 ='2,3,45,0,3,-45,1000,3,0,5000\n' #data string to send to arduino
commandString3 ='3,2,10,0,2,0,1000,2,-10,5000\n' #data string to send to arduino
commandString4 ='4,3,45,0,3,-45,1000,3,0,5000\n' #data string to send to arduino
commandString5 ='S,1,0,0,2,0,2000\n'

############################# SEND DATA ##################################
#function sending data from the PC to the Arduino
def sendData():
    print("Sending command to arduino:")
    #print(commandString)
    ser.write(commandString.encode())
    ser.write(commandString1.encode())
    #ser.write(commandString2.encode())
    #ser.write(commandString3.encode())
    #ser.write(commandString4.encode())
    ser.write(commandString5.encode())

############################# RECEIVE DATA ##################################
    #Function checking if data has been received and if yes reading it
def recData():
    print("Checking for serial input")
    #if data on the serial interface, then read it
    if(ser.in_waiting>0):
        print("Reading serial input")
        ser_input = ser.readline()[:-2] #the last bit gets rid of the new-line chars
        ser_input = ser_input.decode('ascii') #decode input
        print("Data Received from Arduino:")
        print(ser_input) #print received data
    
############################# MAIN ##################################
sendData() #send drive commands once

#continuously check and read serial input
while True:
    current_time=timer() #update current time
    #if interval is reached, check if there is a new serial input
    if(current_time-prev_time>=interval):
        prev_time=current_time
        recData() 