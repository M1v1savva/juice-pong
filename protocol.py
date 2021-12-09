import serial 
from timeit import default_timer as timer

######## INIT serial communication variables################################
ser = serial.Serial('COM4', 57600) #select com-port and the serial com baud rate
ser.flushInput() #empty the serial buffer
ser_input=[] #emtpy the serial input array
current_time = 0 
prev_time = 0
interval = 0.5 #serial read update time

def test_bottom_motor_range():
    commandString0 = '0,' + '0,0,0,' + '0,45,1000,' + '0,-45,1000,' + '0,0,0\n'
    # 0 => 45 => -45 => 0
    ser.write(commandString0)

#Example commands to drive the motors
#commandString0 ='0,0,0,0,1,0,1000,2,0,0,1,90,1000,2,45,0,1,-90,5000,2,-45,0,1,0,0,2,0,0\n'
#commandString1 ='S,0,0,0,1,0,6000,2,0,0\n'

commandString0 ='0,0,0,0,1,0,1000,2,0,0,1,-90,5000,2,-45,0,1,0,0,2,0,0\n'
commandString1 ='S,0,0,0,1,0,1000,2,0,0\n'

#commandString0 ='0,2,45,1000,2,0,1000\n'
#commandString1 ='S,0,0,0,1,0,1000,2,0,0\n'


#commandString0 ='0,1,0,1000,1,-90,1000,1,0,4000\n'
#commandString1 ='S,0,0,0,1,0,1000,2,0,0\n'
#commandString2 ='0,1,0,1000,1,90,1000,1,0,4000\n'
#commandString3 ='S,0,0,0,1,0,1000,2,0,0\n'

#commandString0 = '0,0,0,1000,0, 45,0,1,0,0,2,0,3000,0,0,1000,0,-45,0,1,0,0,2,0,3000\n'
#commandString2 = 'S,0,0,1000,1,0,0,2,0,0\n'

############################# SEND DATA ##################################
#function sending data from the PC to the Arduino
def sendData():
    print("Sending command to arduino:")
    ser.write(commandString0.encode())
    ser.write(commandString1.encode())
    #ser.write(commandString2.encode())
    #ser.write(commandString3.encode())


############################# RECEIVE DATA ##################################
    #Function checking if data has been received and if yes reading it
def recData():
    #print("Checking for serial input")
    #if data on the serial interface, then read it
    if(ser.in_waiting>0):
        print("Reading serial input")
        ser_input = ser.readline()[:-2] #the last bit gets rid of the new-line chars
        ser_input = ser_input.decode('ascii') #decode input
        print("Data Received from Arduino:")
        print(ser_input) #print received data
    
############################# MAIN ##################################
test_bottom_motor_range()

#sendData() #send drive commands once

#continuously check and read serial input
#while True:
#    current_time=timer() #update current time
    #if interval is reached, check if there is a new serial input
#    if(current_time-prev_time>=interval):
#        prev_time=current_time
#        recData() 

