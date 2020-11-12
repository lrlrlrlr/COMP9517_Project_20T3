import os
import cv2
from matplotlib import pyplot as plt
import cv2
import cv2 as cv
import numpy as np
import pandas, seaborn
from collections import defaultdict

# load the files
## Ara2012 - 16 files
## Ara2013 (Canon) - 27 files
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

    # define range of green color in HSV
    lower_green = np.array([28, 84, 50])
    upper_green = np.array([56, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


def show_img(img, title=' '):
    '''
    simply show the image
    :param img:
    :param title:
    :return:
    '''
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

## the most important function of this task!!!
def count_pixels(arr, value=0) -> int:
    '''
    :param arr: flattened array of a single plant in the tray
    :param value: 0 -- black,  255 -- white; default will count the white part of the images.
    :return: INT
    '''
    return len([x for x in arr if x == value])


#
def analysis_threshold_of_medianblur(median2):
    '''
    plot the distribution of the
    :param median2:
    :return:
    '''
    hist = defaultdict(int)
    for i in median2.flatten():
        hist[i] += 1

    df = pandas.DataFrame(hist.items())
    df.columns = ['number', 'freq']

    # remove the 0 's freq
    df_new = df.query('number != 0')
    df_new = df_new.set_index('number')

    seaborn.histplot(df_new, x='number', y='freq')
    plt.show()
    seaborn.lineplot(data=df_new, x='number', y='freq')
    plt.show()




if __name__ == '__main__':
    img = attempt_hsv()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


    ## all the attempts to remove the noise (failed)
    # show_img(gray,'origin - threshed')
    # gau=cv2.GaussianBlur(gray,(5,5),0)
    # show_img(gau, 'after gaussian')
    #
    # box = cv2.boxFilter(gray, -1, (3, 3), normalize=True)
    # show_img(box, 'after box')
    #
    # double = cv2.bilateralFilter(gray, 5, 75, 75)
    # show_img(double, 'after double')



    # naive method to remove the noise: by using medianBlur twice
    median = cv2.medianBlur(gray, 3)
    show_img(median, 'after median ')
    median2 = cv2.medianBlur(median, 3)
    show_img(median2, 'after median twice')


    # rethreshold the image after medianBlur
    # 1. analysis the distribution of grayscale pixels
    # analysis_threshold_of_medianblur(median2)
    # 2. test the best threshold number
    # for i in range(40,85,5):
    #     ret, thresh2 = cv.threshold(median2, i, 255, cv.THRESH_BINARY)
    #     show_img(thresh2, f'thresh2 - {i}')
    #  -- 75 is the best

    _, thresh2 = cv.threshold(median2, 75, 255, cv.THRESH_BINARY)
    show_img(thresh2, f'thresh2 - 75')


    # TODO: we assume we can get the borders of the plants (a,b,c,d), we need to
    # get the array from a,b,c,d
    # arr = GetFlattenArray(a,b,c,d, img)
    # count_pixels(arr)

    # counting the pixels:
    img = cv.imread(r'D:\OneDrive - UNSW\COMP9331\COMP9517_Project_20T3\Tray\Ara2012\ara2012_tray01_fg.png', 0)
    num_labels, labels_im = cv2.connectedComponents(img)

    # we assume there is a plant in y=200~400, x = 500~1000.
    crop_img = img[200:400, 500:1000]
    show_img(crop_img)
    print(f"pixel count: {count_pixels(crop_img.flatten())}")



