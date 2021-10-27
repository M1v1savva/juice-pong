import cv2
import numpy as np
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
    cups_center_point_list = []
    ball_center_point = None
    cv2_im_processed = None
    for frame in frame_list:
        num_cups_detected = 0
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect cups in the image
        cups = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, minDist=50, minRadius=40, maxRadius=50)
        ball = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2.2, minDist=20, minRadius=15, maxRadius=25)
        # ensure at least some cups were found
        if cups is not None:
            # convert the (x, y) coordinates and radius of the cups to integers
            cups = np.round(cups[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the cups
            for (x, y, r) in cups:
                # draw the circle in the output image, then draw a small rectangle at its center
                cups_center_point_list.append((x, y))
                cv2.circle(output, (x, y), r, (255, 0, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            num_cups_detected = len(cups)
            # cv2.waitKey(0)

        if ball is not None:
            ball = np.round(ball[0, :]).astype("int")
            for (x, y, r) in ball:
                ball_center_point = (x, y)
                cv2.circle(output, (x, y), r, (0, 255, 0), 3)

        # Convert to PIL Image
        cv2_im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        draw = ImageDraw.Draw(pil_im)

        # Choose a font
        font = ImageFont.truetype('data/font/Roboto-Bold.ttf', 55)

        # Draw the text
        draw.text((20, 20), 'Number of cups detected = ' + str(num_cups_detected), font=font, fill="#000000")
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        if cv2_im_processed is not None:
            frame_list_contours.append(cv2_im_processed)
        else:
            frame_list_contours.append(output)
        count += 1

    fps = 30
    height, width, layers = frame_list[0].shape
    size = (width, height)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_list_contours)):
        out.write(frame_list_contours[i])
    out.release()
    print('successfully saved:', path_out)


if __name__ == "__main__":
    detect_cups('data/input/cups_video1.mov', 'data/output/compiled_cups_video1.mp4')
