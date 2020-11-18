import cv2
import numpy


def nothing(x):
    pass

# ignore video capture for now, this script will focus on static images
# cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

# Set initial values for Trackbars
cv2.createTrackbar("LH", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("LS", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("UH", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("MNSZ", "Trackbars", 0, 500, nothing)
cv2.createTrackbar("MXSZ", "Trackbars", 500, 500, nothing)

# Set multiplier for MNSZ/MXSZ (min and max threshholds)
multiplier = 5


while True:
    # ignore video capture for now, this script will focus on static images
    # _, frame = cap.read()
    frame = cv2.imread(r'D:\OneDrive - UNSW\COMP9331\COMP9517_Project_20T3\Tray\Ara2012\ara2012_tray01_rgb.png')
    x,y,_ = frame.shape
    frame = cv2.resize(frame, (int(x*0.2), int(y*0.2)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Store trackbar variables
    lower_hue = cv2.getTrackbarPos("LH", "Trackbars")
    lower_saturation = cv2.getTrackbarPos("LS", "Trackbars")
    lower_value = cv2.getTrackbarPos("LV", "Trackbars")
    upper_hue = cv2.getTrackbarPos("UH", "Trackbars")
    upper_saturation = cv2.getTrackbarPos("US", "Trackbars")
    upper_value = cv2.getTrackbarPos("UV", "Trackbars")
    min_size = cv2.getTrackbarPos("MNSZ", "Trackbars") * multiplier
    max_size = cv2.getTrackbarPos("MXSZ", "Trackbars") * multiplier

    lower_hsv_range = numpy.array([lower_hue, lower_saturation, lower_value])
    upper_hsv_range = numpy.array([upper_hue, upper_saturation, upper_value])
    mask = cv2.inRange(hsv, lower_hsv_range, upper_hsv_range)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if min_size <= cv2.contourArea(c) <= max_size]
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Show the frame, mask and result
    cv2.imshow("frame", frame)

    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    # Break out of this while loop on 'escape' press
    key = cv2.waitKey(1)
    if key == 27:
        break

# ignore video capture for now, this script will focus on static images
# cap.release()
cv2.destroyAllWindows()