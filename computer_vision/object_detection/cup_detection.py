import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def hough_transform(frame):
    cups_center_point_list = []
    num_cups_detected = 0

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('data/output/frame.png', frame)
    # cv2.imwrite('data/output/gray.png', gray)

    # detect cups in the image
    cups = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.8, minDist=40, minRadius=50, maxRadius=60)
    # ensure at least some cups were found
    if cups is not None:
        # convert the (x, y) coordinates and radius of the cups to integers
        cups = np.round(cups[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the cups
        for (x, y, r) in cups:
            # draw the circle in the output image, then draw a small rectangle at its center
            cups_center_point_list.append((x, y))
            # cv2.circle(output, (x, y), r, (0, 0, 255), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        num_cups_detected = len(cups)

    # # Convert to PIL Image
    # cv2_im_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # pil_im = Image.fromarray(cv2_im_rgb)
    # draw = ImageDraw.Draw(pil_im)
    #
    # # Choose a font
    # font = ImageFont.truetype('data/font/Roboto-Bold.ttf', 55)
    #
    # # Draw the text
    # draw.text((20, 20), 'Number of cups detected = ' + str(num_cups_detected), font=font, fill="#FFFFFF")
    # cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    # cv2.imwrite(path_out, cv2_im_processed)
    # print('successfully saved:', path_out)

    return output, cups_center_point_list

