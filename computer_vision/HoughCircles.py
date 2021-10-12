import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image


def detect_cups(path_in, path_out):
    vid = cv2.VideoCapture(path_in)
    frame_list = []
    success, image = vid.read()
    while success:
        frame_list.append(image)
        success, image = vid.read()

    count = 0
    frame_list_contours = []
    center_point_list = []
    for frame in frame_list:
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, minDist=100, minRadius=50, maxRadius=100)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (255, 0, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # cv2.waitKey(0)

            # Convert to PIL Image
            cv2_im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)

            draw = ImageDraw.Draw(pil_im)

            # Choose a font
            font = ImageFont.truetype('data/font/Roboto-Bold.ttf', 55)

            # Draw the text
            draw.text((20, 20), 'Number of cups detected = ' + str(len(circles)), font=font)
            cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        frame_list_contours.append(cv2_im_processed)
        count += 1

    fps = 30
    height, width, layers = frame_list_contours[0].shape
    size = (width, height)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_list_contours)):
        out.write(frame_list_contours[i])
    out.release()
    print('successfully saved:', path_out)


if __name__ == "__main__":
    detect_cups('data/input/cups_video.mp4', 'data/output/compiled_video.mp4')
