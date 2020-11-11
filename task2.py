import os
import cv2
from matplotlib import pyplot as plt
import cv2
import numpy as np


# load the files
## Ara2012 - 16 files
## Ara2013 (Canon) - 27 files
def get_files_path():
    # ara2012_rgb = ['Tray/Ara2012/'+file for file in  os.listdir('Tray/Ara2012') if '_rgb.png' in file]
    # ara2012_fg = ['Tray/Ara2012/'+file for file in  os.listdir('Tray/Ara2012') if '_fg.png' in file]
    # ara2013_rgb = ['Tray/Ara2013-Canon/'+file for file in  os.listdir('Tray/Ara2012') if '_rgb.png' in file]
    # ara2012_fg = ['Tray/Ara2012/'+file for file in  os.listdir('Tray/Ara2012') if '_fg.png' in file]
    pass


image = cv2.imread(r'D:\OneDrive - UNSW\COMP9331\COMP9517_Project_20T3\Tray\Ara2012\ara2012_tray01_rgb.png')
image = cv2.resize(image, None, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)


def attempt_hsv():
    '''
    Use hsv to parse the image
    ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
    :return: image array
    '''
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([28, 84, 50])
    upper_blue = np.array([56, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # cv2.imshow('frame',frame)
    plt.imshow(mask)
    plt.show()
    plt.imshow(res)
    plt.show()

    return res


if __name__ == '__main__':
    attempt_hsv()