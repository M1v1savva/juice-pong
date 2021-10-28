import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def grab_frame(cap):
    ret, frame = cap.read()
    return cv2.cvtColor(frame, 0)


def detect_cups(path_out):
    frame = cv2.VideoCapture(0)
    frame = grab_frame(frame)

    cups_center_point_list = []
    ball_center_point = ()
    num_cups_detected = 0

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('data/output/frame4.png', frame)

    # cv2.imwrite('data/output/gray.png', gray)
    # cv2.imwrite('data/output/gray1.png', cv2.bitwise_not(gray))

    # cv2.imwrite('data/output/canny.png', cv2.Canny(frame, threshold1=300, threshold2=300))

    # detect cups in the image
    cups = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, minDist=40, minRadius=40, maxRadius=50)
    ball = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, minDist=10, minRadius=10, maxRadius=22)
    # ensure at least some cups were found
    if cups is not None:
        # convert the (x, y) coordinates and radius of the cups to integers
        cups = np.round(cups[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the cups
        for (x, y, r) in cups:
            # draw the circle in the output image, then draw a small rectangle at its center
            cups_center_point_list.append((x, y))
            cv2.circle(output, (x, y), r, (0, 0, 255), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1)
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
    draw.text((20, 20), 'Number of cups detected = ' + str(num_cups_detected), font=font, fill="#FFFFFF")
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    cv2.imwrite(path_out, cv2_im_processed)
    print('successfully saved:', path_out)

    print('-----')
    print('ball coordinate =', ball_center_point)
    print('cups coordinates =', cups_center_point_list)
    print('Number of cups detected =', num_cups_detected)
    print('-----')


if __name__ == "__main__":
    detect_cups('data/output/photo.png')
