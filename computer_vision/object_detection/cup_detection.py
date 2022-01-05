import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def hough_transform(frame):
    cups_center_point_list = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect cups in the image
    cups = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, minDist=22, minRadius=45, maxRadius=55)
    # ensure at least some cups were found
    if cups is not None:
        # convert the (x, y) coordinates and radius of the cups to integers
        cups = np.round(cups[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the cups
        for (x, y, r) in cups:
            # draw the circle in the output image, then draw a small rectangle at its center
            cups_center_point_list.append((x, y))
            # cv2.circle(output, (x, y), r, (0, 0, 255), 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
    
    return cups_center_point_list

