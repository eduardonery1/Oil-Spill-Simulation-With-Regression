# -*- coding: utf-8 -*-

from sim import *
import time
import cv2
import numpy as np
from math import sqrt

F = []
M = []
GAMMA = []

def preprossImg(image, resolution):
    img = np.array(image,dtype=np.uint8)
    img.resize([resolution[1],resolution[0],3])
    img = cv2.flip(img, 0)
    img = cv2.inRange(img, (0,0,0), (50,50,50))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

def saveMassCenters(contour):
    moments = cv2.moments(contour)
    if not (moments['m00'] <= 300000): return -1
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    mass_center = (cx, cy)
    M.append(mass_center)
    return 0


simxFinish(-1)

clientID = simxStart('127.0.0.1', 19997, True, True, 5000, 5)
simxStartSimulation(clientID, simx_opmode_oneshot)


if clientID!=-1:
    res, v1 = simxGetObjectHandle(clientID, 'kinect_rgb', simx_opmode_oneshot_wait)
    err, resolution, image = simxGetVisionSensorImage(clientID, v1, 0, simx_opmode_streaming)
    kernel = np.zeros((3,3), 'uint8')

    while (simxGetConnectionId(clientID) != -1):
        err, resolution, image = simxGetVisionSensorImage(clientID, v1, 0, simx_opmode_buffer)

        if err == simx_return_ok:
            img = preprossImg(image, resolution)
            F.append(img)

            contornos, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if -1 == saveMassCenters(contornos[0]): break

            elipse = cv2.fitEllipse(contornos[0])
            GAMMA.append(elipse)

        elif err == simx_return_novalue_flag:
            pass

        else:
          print("Could not get image sensor")
else:
  print ("Failed to connect to remote API Server")
  simxFinish(clientID)

cv2.destroyAllWindows()

#-- Calculate DeltaY and distance
deltaY = M[-1][1] - M[0][1]
distance = sqrt((M[-1][0] - M[0][0])**2 + (M[-1][1] - M[0][1])**2)

#-- Draw mass centers and elipses while calculating the total mean Area of the ellipses
result = cv2.imread('graph.png')
Am = 0
for g in GAMMA:
    cv2.ellipse(result, g, (255, 0, 255), -1)
    Am += 2*3.14159272*g[1][0]*g[1][1]
Am /= len(GAMMA)
for centroid in M:
    cv2.circle(result, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
cv2.imwrite('ans.png', result)

#-- Linear Regression model to predict Affected Areas based on deltaY, distance and mean Ellipse area
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
dataset = pd.read_csv('data.csv')
X = dataset.drop('Affected area', axis=1)
y = dataset['Affected area']
reg = LinearRegression().fit(X, y)

#-- Write Coefficients of the model
f = open('modelo.txt', 'w')
f.write(str(reg.coef_))
f.close()
