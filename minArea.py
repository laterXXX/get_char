import cv2
import numpy as np
import os

# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white
# wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg

files = os.listdir('./rect_license/')
for file_name in files:

    img = cv2.pyrDown(cv2.imread('./rect_license/'+file_name, cv2.IMREAD_UNCHANGED))

    cv2.imshow('src',img)
    cv2.waitKey(0)
    img= cv2.resize(img,(1200,400))
    # threshold image
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    127, 255, cv2.THRESH_BINARY)
    # find contours and get the external one

    cv2.imshow('threshed_img',threshed_img)
    cv2.waitKey(0)
    threshed_img = cv2.bitwise_not(threshed_img)

    # contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    res = threshed_img
    res[:] = 255

    cv2.drawContours(res,contours,-1,(0,255,15),2)
    cv2.imshow('drawContours',res)
    cmin = 100
    cmax = 1500
    cv2.waitKey(0)
    for c in contours:
        if c.size > cmax or c.size < cmin:
            contours.remove(c)
    cv2.drawContours(res,contours,-1,(0,255,15),2)
    cv2.imshow('remove',res)
    cv2.waitKey(0)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))

        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        # and draw the circle in blue
        img = cv2.circle(img, center, radius, (255, 0, 0), 2)

    print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

    cv2.imshow("contours", img)


    while True:
        key = cv2.waitKey(1)
        if key == 27: #ESC key to break
            break

    cv2.destroyAllWindows()