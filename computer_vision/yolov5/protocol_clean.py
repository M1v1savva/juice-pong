import serial
from timeit import default_timer as timer

######## INIT serial communication variables################################
ser = serial.Serial('/dev/ttyACM0', 57600) # select com-port and the serial com baud rate
ser.flushInput()  # empty the serial buffer

def test_bottom_motor_range():
    commandString0 = '0,' + '0,0,1000,' + '0,45,1000,' + '0,-45,1000,' + '0,0,1000\n'
    # initial delay 1000 to avoid overhead
    # 0 => 45 => -45 => 0
    #requires 1 second delay after each rotation feed
    #otherwise previous feed gets ignored and MC jumps to the latest rotation
    commandString1 = 'S,' + '0,0,1000\n'
    #stops the motor

    ser.write(commandString0.encode())
    ser.write(commandString1.encode())

def test_middle_motor_range():
    commandString0 = '0,' + '2,0,1000,' + '2,45,1000,' + '2,-45,1000,' + '2,0,1000\n'
    # initial delay 1000 to avoid overhead
    # 0 => 45 => -45 => 0
    #requires 1 second delay after each rotation feed
    #otherwise previous feed gets ignored and MC jumps to the latest rotation
    commandString1 = 'S,' + '2,0,1000\n'
    #stops the motor

    ser.write(commandString0.encode())
    ser.write(commandString1.encode())

def test_top_motor_range():
    commandString0 = '0,' + '1,0,1000,' + '1,90,1000,' + '1,-90,1000,' + '1,0,1000\n'
    # initial delay 1000 to avoid overhead
    # 0 => 90 => -90 => 0
    #requires 1 second delay after each rotation feed
    #otherwise previous feed gets ignored and MC jumps to the latest rotation
    commandString1 = 'S,' + '1,0,1000\n'
    #stops the motor

    ser.write(commandString0.encode())
    ser.write(commandString1.encode())

def default_shot(bot_angle = 0, wait=5000,delay=0):
    print('shot config being transfered to MC')
    commandString0 = '0,' + '0,0,1000,' + '1,0,0,' + '2,0,0,'
    # feed rubbish first to avoid overhead
    commandString0 += ('0,' + str(bot_angle) + ',1000,') + '2,45,1000,' + '1,90,1000,'
    # new "block", hence, need delay for the first motor rotation to avoid movement cancellation
    # additionally, delays for the rest of the motors for smoother movement (doesnt need to be rapid)
    # moving to starting position
    commandString0 += ('2,0,' + str(wait) + ',') + ('1,0,' + str(delay) + '\n')
    # new "block", hence, need delay for the first motor rotation to avoid movement cancellation
    # + need to wait before executing the shot to place the ball
    # executing the shot
    commandString1 = 'S,' + '0,0,1000,' + '1,0,0,' + '2,0,0\n'
    # stops the motor
    ser.write(commandString0.encode())
    ser.write(commandString1.encode())

############################# MAIN ##################################

# motor angles:
# positive angles for mid and top motors throw execution
# bot motor: positive angle to the left, negative to the right (looking from the cups towards the setup)
default_shot(bot_angle=-2, delay=200, wait=5000)

####################### LOOK-UP TABLE ###############################
#
# PAIRS OF VALUES (bot_angle, top_motor_delay)
# LEFT TO RIGHT, BOTTOM:
# (7,50), (0,50), (-7,50), (-13,50)
# LEFT TO RIGHT, BOT MID:
# (4,135), (-3,135), (-10,135)
# LEFT TO RIGHT, TOP MID:
# (2,185), (-5,185)
# TOP:
# (-3,200)
