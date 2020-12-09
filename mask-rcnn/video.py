#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from mask_rcnn import MASK_RCNN 
from PIL import Image
import numpy as np
import cv2
import time

mask_rcnn = MASK_RCNN()
# Call camera
capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")

fps = 0.0
while(True):
    t1 = time.time()
    # Read a frame
    ref,frame=capture.read()
    # Format conversion, BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # Turn into Image
    frame = Image.fromarray(np.uint8(frame))

    # Test
    frame = np.array(mask_rcnn.detect_image(frame))

    # RGBtoBGR meets the opencv display format
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)

    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break
