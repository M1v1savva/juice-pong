//EDMO-Setups: Serial communication & basic motor control
//Author: Lucas Dahl
//Date:23/09/2021
/*
      __
   __/ o \           - -  -
  (__     \______
     |    __)   /  - -  - - -
      \________/
         _/_ _\_    - - -

10:

0,2,45,0,1,-90,0
0,2,0,0,1,0,0

9:

0,2,45,0,1,-90,0
0,2,0,0,1,0,150

8:

0,2,45,0,1,-90,0
0,2,0,0,1,0,180

7:

0,2,45,0,1,-90,0
0,2,0,0,1,0,210

6: 

0,2,45,0,1,-90,0
0,2,0,0,1,0,230

5: 

0,2,45,0,1,-90,0
0,2,0,0,1,0,250

0,2,45,0,1,-90,0,2,0,5000,1,0,0

*/ 
//Description: 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Please enter our specific servo limits for servo 1,2,3 and 4 here
//You can find the corresponding limits in your EDMO box

//105 560
//198 380
int PPMMIN[]  {187, 98, 200, 0}; //PLEASE ENTER: The lower motor PPM limit (Servo Min) as noted in your EDMO-setup
int PPMMAX[]  {386, 490, 370, 0}; //PLEASE ENTER: The upper motor PPM limit (Servo Max) as noted in your EDMO-setup

int ANGMIN[] {-45, -90, -45, -45}; //Minimum motor range in degs
int ANGMAX[] {45, 90, 45, 45}; //Maximum motor range in degs

int POTIMIN[]  {585, 520, 573, 0}; //PLEASE ENTER: The lower potentiometer limit (Poti Low)as noted in your EDMO-setup
int POTIMAX[]  {315, 251, 291, 0}; //PLEASE ENTER: The lupper potentiometer limit (Poti High)as noted in your EDMO-setup
int POTIPINS[] = {14, 15, 16, 17}; //analog pins to which the potis' of the servos are connected (A0=14, A1=15, A2=16, ...)

//Motor Variables
int PPM=0;

#include <Adafruit_PWMServoDriver.h>
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
const unsigned int NUM_MOTORS = 4; //number of used motors

//Buffer Management
const int bufferLength=25; //buffer size
const int commandLength=100; //maximum character count of command sequences received via the serial
int inputCount[bufferLength]; //counter for the number of received inputs (motor number, angle, delay)x X  (not including the command ID)
int controlSeq[bufferLength][commandLength]; //array for storing all received input command sequences
int controlId[bufferLength]; //array for storing the IDs of the received input commands
int writeIdx =0; 
int driveIdx =0;
unsigned int driveCounter =0;
unsigned int readCounter =0;
int pkg_idx =0;
boolean startFlag = false;
boolean lapEvent = false;
boolean overflow =false;
boolean firstIter =true;
int stopFlag[bufferLength];

//Communication variables
char record[commandLength];
int charCount = 0; //storing the number of characters received in one command sequence. Required to check that it is not exceeding the maximum 'commandLength'
char recvchar; //required for parsing the input string
byte indx = 0; //required for parsing the input string
boolean idInput = false; //flag indicating that the ID of a command has been read

//Timing Variables
unsigned long prev1Millis = 0; //timer for reading the serial input
unsigned long prev2Millis = 0; //timer for driving the motors
unsigned long startTime = 0; //required for timing the drive delays received in a command sequence
int delayTime =0; //intermediate storage for the received movement delay times
const int INTERVAL = 10; //update interval in [ms] of the reading and driving loops

// *********************************************************************************************************************** //
void setup() {
  // put your setup code here, to run once:
  Serial.begin(57600);  // start serial com with baud rate of 57600
  ///////Motor control/////////////////////////////////
  pwm.begin();
  pwm.setPWMFreq(50);  // Analog servos run at ~60 Hz update, we need 50Hz
  //
  while(!Serial); //wait for serial cable and receiver to be connected
  prev1Millis = millis();
  prev2Millis = millis()+5; //the motor drive loop is updated with a 5ms offset to the receive loop
}

// *********************************************************************************************************************** //
void loop() {
  //receive loop
  if((millis()-prev1Millis)>=INTERVAL){
    prev1Millis = millis();
    //check if we have a buffer overflow. If not read the input. If yes do nothing until buffer is fine again
    if(overflow==false){
      readInput();//check if there is data on the serial input. If yes it parses the input
    }else{
      //Serial.println("!!!!!!!!Buffer Overflow!!!!!!!!!!!");
      //Do not read the serial when buffer is overflowing
    }
  }
  //drive loop
  if((millis()-prev2Millis)>=INTERVAL){
    prev2Millis = millis();
    driveMotor(); //drive motors
  }
}
// *********************************************************************************************************************** //
////read input received via serial input//////////////
void readInput(){
  //Check for data on the serial input and read input data if availble
  if (Serial.available()){    
    recvchar = Serial.read();
    if (recvchar != '\n')//read data until you read a newline (enter)
    { 
        record[indx++] = recvchar; //store received input characters in 'record'
    }
    else if (recvchar == '\n') //if newline character is read
    {
      record[indx] = '\0';
      charCount = indx; //store number of received characters
      indx = 0; //set index to zero
      //if the first recieved char is 'S' the stop all motors
      if(record[0]=='S'){       
        stopFlag[writeIdx]=1;
        convertData(record); //fct. to parse the serial input data stored in record
        memset(record, 0, sizeof(record)); //empty record array 
        recvchar=' '; //empty input char
      //if no stop command is received, parse the input
      }else{
        convertData(record); //fct. to parse the serial input data stored in record
        memset(record, 0, sizeof(record)); //empty record array 
        recvchar=' '; //empty input char
      }
    }
  }
}
// *********************************************************************************************************************** //
////Parse data received via serial input//////////////
void convertData(char record[])
{   
    int i = 0;
    char *index = strtok(record, ","); //separate string at comma and save input before the comma
    while(index != NULL)
    {
      //if not the first command of the input sequence write the different inputs (motor#, angle, delay) to 'controlSeq'
      if(idInput==true){
         controlSeq[writeIdx][i++] = atof(index); //convert the different input elements to float and save them
         index = strtok(NULL, ","); //separate string at next comma and save input before the comma
         inputCount[writeIdx]=inputCount[writeIdx]+1; //count the number of received input elements/commands
      //if the first command of the input sequence, it is the command string ID that needs to be saved in 'controlID'
      }else if(idInput==false){
        controlId[writeIdx]= atof(index); 
        index = strtok(NULL, ",");
        idInput=true;
      }
    }
    idInput =false;
    //Input verification: Check if the received input (without ID) is not empty, a multiple of 3, and the length of the input command is not longer than the max 'commandLength'
    if(inputCount[writeIdx]>0 && inputCount[writeIdx]%3 == 0 && charCount <= commandLength){
      writeIdx = (writeIdx+1)% bufferLength; //increment buffer. Fill buffer until buffer length is reached. Then start from zero
      readCounter = readCounter +1; //counter for recieved commands to make sure that we are not driving more commands than received
      firstIter = false;
    //if the input data are invalid reset the input counter and send a message
    }else{
      inputCount[writeIdx]=0;
      Serial.println("Invalid Input");
    }
    //when the buffer array is full and starts to be filled from the beginning set a flag to prevent that we overwrite commands that
    //we overwrite commands that have not yet been executed (overlapping event)
    if(readCounter%bufferLength==0 && firstIter==false){
      //Serial.println("overlapping event");
      lapEvent=true;
    }
}

// *********************************************************************************************************************** //
////Drive motors according to the received inputs//////////////
void driveMotor(){
  //check if the write cycle is already one 'lap' further than the drive cycle. If not then do:
  if(lapEvent == false){
    //make sure that the driving actions are not 'overtaking' the receiving actions
    if(driveIdx < writeIdx){
      overflow =false; //reset overflow indicator if set
      //now process one data package (motor#,angle,delay) after the other until the complete command string has been processed
      if(pkg_idx < inputCount[driveIdx]){
        delayTime = controlSeq[driveIdx][pkg_idx+2];//store the delay time of the corresponding data package
        //save the startTime
        if(startFlag == false){
          startTime = millis();
          startFlag = true;
        }
        //when the delay time of the corresponding package has passed drive the motor of the same data package
        //Here we distinguish between drive and stop actions
        //1. Drive Actions
        if((millis()-startTime >= delayTime)&& stopFlag[driveIdx] == 0){
          PPM = map(controlSeq[driveIdx][pkg_idx+1],ANGMIN[controlSeq[driveIdx][pkg_idx]],ANGMAX[controlSeq[driveIdx][pkg_idx]],PPMMIN[controlSeq[driveIdx][pkg_idx]],PPMMAX[controlSeq[driveIdx][pkg_idx]]);//MAP input angle from physical servo range to PPM range
          PPM = constrain(PPM,PPMMIN[controlSeq[driveIdx][pkg_idx]],PPMMAX[controlSeq[driveIdx][pkg_idx]]); //constrain PPM values to min and max values. DO NOT MODIFY OR (RE)MOVE THIS LINE!!!
          pwm.setPWM(controlSeq[driveIdx][pkg_idx], 0,PPM); //send PPM values to motors
          pkg_idx = pkg_idx+3; //proceed to the next data package of this command string
          startFlag = false; //reset the start flag for the next delay time of the next data package
        //2. Stop Actions
        }else if((millis()-startTime >= delayTime)&& stopFlag[driveIdx] == 1){
          pwm.setPWM(controlSeq[driveIdx][pkg_idx], 0,0); //stop motor movement
          pkg_idx = pkg_idx+3; //proceed to the next data package of this command string
          startFlag = false; //reset the start flag for the next delay time of the next data package
        }
      //if all data packages of one command sequence have been processed send the processed commandseq ID to the PC and go to the next received command
      }else{
        //send ID of the processed command sequence
        //In case it was a normal drive action send the ID of the processed command sequence
        if(stopFlag[driveIdx] == 0){
          Serial.print("ID:");
          Serial.println(controlId[driveIdx]);
        //In case it was a stop action send 'S' as a feedback to the PC
        }else if(stopFlag[driveIdx] == 1){
          Serial.println("ID: S");
        }
        
        driveIdx = (driveIdx+1)% bufferLength; //increment the index of the command sequence to drive. Fill buffer until buffer length is reached. Then start from zero
        driveCounter = driveCounter +1; //counter for drive events to prevent that driving events 'overtake' the reading events
        pkg_idx = 0;
        //keep track if driving buffer is full and starts again from zero. In this case reset the 'overlapping' indicator
        if(driveCounter%bufferLength==0){
          //Serial.println("overlapping event RESET");
          lapEvent=false;
        }
      }
    }
  //check if the write cycle is already one 'lap' further than the drive cycle. If yes then do:
  }else if(lapEvent == true){
    //make sure that the write index does not overtake the drive index and starts overwriting drive commands that have not yet been 
    //executed
    if(writeIdx < driveIdx){
      overflow =false;
      if(pkg_idx < inputCount[driveIdx]){
        delayTime = controlSeq[driveIdx][pkg_idx+2];
        if(startFlag == false){
          startTime = millis();
          startFlag = true;
        }
        if((millis()-startTime >= delayTime)&& stopFlag[driveIdx] == 0){
          PPM = map(controlSeq[driveIdx][pkg_idx+1],ANGMIN[controlSeq[driveIdx][pkg_idx]],ANGMAX[controlSeq[driveIdx][pkg_idx]],PPMMIN[controlSeq[driveIdx][pkg_idx]],PPMMAX[controlSeq[driveIdx][pkg_idx]]);//MAP input angle from physical servo range to PPM range
          PPM = constrain(PPM,PPMMIN[controlSeq[driveIdx][pkg_idx]],PPMMAX[controlSeq[driveIdx][pkg_idx]]); //constrain PPM values to min and max values. DO NOT MODIFY OR (RE)MOVE THIS LINE!!!
          pwm.setPWM(controlSeq[driveIdx][pkg_idx], 0,PPM); //send PPM values to motors
          pkg_idx = pkg_idx+3;
          startFlag = false;
        }else if((millis()-startTime >= delayTime)&& stopFlag[driveIdx] == 1){
          pwm.setPWM(controlSeq[driveIdx][pkg_idx], 0,0); //stop motor movement
          pkg_idx = pkg_idx+3; //proceed to the next data package of this command string
          startFlag = false; //reset the start flag for the next delay time of the next data package
        }
      }else{
        //send ID of the processed command sequence
        //In case it was a normal drive action send the ID of the processed command sequence
        if(stopFlag[driveIdx] == 0){
          Serial.print("ID:");
          Serial.println(controlId[driveIdx]);
        //In case it was a stop action send 'S' as a feedback to the PC
        }else if(stopFlag[driveIdx] == 1){
          Serial.println("ID: S");
          //stopFlag =false;
        }
        
        driveIdx = (driveIdx+1)% bufferLength; //fill buffer until buffer length is reached. Then start from zero
        driveCounter = driveCounter +1;
        pkg_idx = 0;
        if(driveCounter%bufferLength==0){
//        Serial.println("overlapping event RESET");
          lapEvent=false;
        }
      }
    //If the write cyle overtakes the drive cycle that means that the buffer is overflowing. Set the overflow flag that prevents that
    //further commands can be received and written to the buffer. Also send a warning to the PC.
    }else{
//    Serial.print("Buffer Overflow!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      overflow =true; 
      Serial.println("Buffer overflow!");
      //execute the next drive command to free some buffer
      if(pkg_idx < inputCount[driveIdx]){
        delayTime = controlSeq[driveIdx][pkg_idx+2];
        if(startFlag == false){
          startTime = millis();
          startFlag = true;
        }
        if((millis()-startTime >= delayTime)&& stopFlag[driveIdx] == 0){
          PPM = map(controlSeq[driveIdx][pkg_idx+1],ANGMIN[controlSeq[driveIdx][pkg_idx]],ANGMAX[controlSeq[driveIdx][pkg_idx]],PPMMIN[controlSeq[driveIdx][pkg_idx]],PPMMAX[controlSeq[driveIdx][pkg_idx]]);//MAP input angle from physical servo range to PPM range
          PPM = constrain(PPM,PPMMIN[controlSeq[driveIdx][pkg_idx]],PPMMAX[controlSeq[driveIdx][pkg_idx]]); //constrain PPM values to min and max values. DO NOT MODIFY OR (RE)MOVE THIS LINE!!!
          pwm.setPWM(controlSeq[driveIdx][pkg_idx], 0,PPM); //send PPM values to motors
          pkg_idx = pkg_idx+3;
          startFlag = false;
        }else if((millis()-startTime >= delayTime)&& stopFlag[driveIdx] == 1){
          pwm.setPWM(controlSeq[driveIdx][pkg_idx], 0,0); //stop motor movement
          pkg_idx = pkg_idx+3; //proceed to the next data package of this command string
          startFlag = false; //reset the start flag for the next delay time of the next data package
        }
      }else{
        //send ID of the processed command sequence
        //In case it was a normal drive action send the ID of the processed command sequence
        if(stopFlag[driveIdx] == 0){
          Serial.print("ID:");
          Serial.println(controlId[driveIdx]);
        //In case it was a stop action send 'S' as a feedback to the PC
        }else if(stopFlag[driveIdx] == 1){
          Serial.println("ID: S");
          //stopFlag =false;
        }

        driveIdx = (driveIdx+1)% bufferLength; //fill buffer until buffer length is reached. Then start from zero
        driveCounter = driveCounter +1;
        overflow =false;
        pkg_idx = 0;
        if(driveCounter%bufferLength==0){
//          Serial.println("overlapping event RESET");
          lapEvent=false;
          //delay(1000);
        }
      }
    }
  }
}
// *********************************************************************************************************************** //
//Stop all motor movements. All motors are switched off
void stopAll(){
  for(int i=0;i<NUM_MOTORS;i++){
    pwm.setPWM(i, 0,0); //stop motor movement
  }
}
