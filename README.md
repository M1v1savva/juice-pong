# Maastricht University - Project 3-1 
## Water pong - Throwing Robot 
### Computer Vision

The Circle Hough Tranform (CHT) feature extraction technique was used to detect the cups (circles) on the water pong game board in our case. 

To perform your own videos circles detection:

1. Make sure you already have the cv2 (OpenCV) library installed on your machine.
2. Upload an mp4 file inside the computer_vision/data/input folder.
3. Run the detect_cups python script inside HoughCircle.py file with the right input and output paths.
4. Wait for the algorithm to be done.
5. Enjoy your video!

--> You may need to tune the minRadius and maxRadius of the cv2.HoughCircles parameters (l. 23) to detect circles of the size you need (in pixels).
