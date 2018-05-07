#!/usr/bin/env python3

import random
import cv2
import numpy as np

def vision(im, camera_matrix, dist_coeffs):
  imdbg = im.copy()

  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  h,w,_ = hsv.shape
  hsv = cv2.resize(hsv, (int(w/4),int(h/4)))

  polys = []
  # orange
  polys.extend(detect_by_hue('o',hsv,imdbg,0,20))
  # green
  polys.extend(detect_by_hue('g',hsv,imdbg,60,70))
  # yellow
  polys.extend(detect_by_hue('y',hsv,imdbg,25,50))
  # blue
  polys.extend(detect_by_hue('b',hsv,imdbg,100,140))

  a = 0.058
  obj_points = np.array([
      [[-a/2],[-a/2],[0]],
      [[a/2],[-a/2],[0]],
      [[a/2],[a/2],[0]],
      [[-a/2],[a/2],[0]],
      ])

  # cube position in camera space
  if len(polys) > 0:
    for color,poly in polys:
      
      image_points = []
      for point in poly:
        image_points.append(4*np.transpose(point))
      image_points = np.array(image_points).astype(float)
      
      if len(image_points) == 4:
        _,rvec,tvec = cv2.solvePnP(obj_points,image_points,camera_matrix,dist_coeffs)
 
        tvcam,rvcam = np.array([[-0.057],[ 0.015],[0.413]]),np.array([[2.040],[0.970],[-0.408]])
        rmcam,_ = cv2.Rodrigues(rvcam)
        
        x,y,z = np.dot(-np.transpose(rmcam), tvcam-tvec)
        x,y,z = [int(1000*v) for v in (x,y,z)]
        cx,cy = [int(c) for c in np.transpose(image_points[0])[0]]
        cv2.putText(imdbg, "%s %d %d %d"%(color,x,y,z), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),3)

  return imdbg

def detect_by_hue(color,hsv,imdbg,hmin, hmax):
  h,w,_ = hsv.shape
  smask = np.zeros((h,w,1),np.uint8)

  dh = 25
  polygons = []
  for v in range(100,255,dh):
    for s in range(100,255,dh):
      lower = np.array([hmin,s,v])
      upper = np.array([hmax,250,250])
      mask = cv2.inRange(hsv,lower,upper)
      
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
      mask = cv2.dilate(mask, kernel)
      mask = cv2.erode(mask, kernel)

      _,contours,_ = cv2.findContours(mask, 1, 2)
      contours = sorted(contours, key=cv2.contourArea, reverse=True)
      for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt,True), True)

        is_convex = cv2.isContourConvex(approx)
        area = cv2.contourArea(approx)
        if 0.20*h*w > area and area > 0.02*h*w:
          n = len(approx)
          p = approx[:,0,:]
          if is_convex and n == 4:

            def sqdist(s):
              return s[0]*s[0] + s[1]*s[1]

            s01 = p[0]-p[1]
            s12 = p[1]-p[2]
            s23 = p[2]-p[3]
            s30 = p[3]-p[0]

            dl = np.abs(sqdist(s01) - sqdist(s23))
            dr = np.abs(sqdist(s12) - sqdist(s30))
            a = np.abs(np.dot(s01,s23))
            b = np.abs(np.dot(s30,s12))
            if a < 10000 and b < 10000 and dl <1000 and dr<1000:
              m = np.zeros((h,w,1),np.uint8)
              cv2.fillPoly(m, [p], (5*dh,))
              smask = cv2.add(smask,m)
    
  #cv2.imshow("smask-%d-%d"%(hmin,hmax),smask)
  smask = cv2.inRange(smask, (100,), (255,))

  #cv2.imshow("tsmask-%d-%d"%(hmin,hmax),smask)
  _,contours,_ = cv2.findContours(smask, 1, 2)

  polys = []
  for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt,True), True)
    cv2.drawContours(imdbg, [4*approx], -1, (255,0,0), 2)
    polys.append((color,approx))
  
  return polys

def main_video():

  cal_params = cv2.FileStorage("/home/jdam/jevois_640.txt", cv2.FILE_STORAGE_READ)
  camera_matrix = cal_params.getNode("cameraMatrix").mat()
  dist_coeffs = cal_params.getNode("dist_coeffs").mat()

  cap = cv2.VideoCapture(1)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);

  while True:
    r,im = cap.read()

    imdbg = vision(im, camera_matrix, dist_coeffs)
    cv2.imshow("",imdbg)
    key = cv2.waitKey(10)
    if key == 27:
      break

  cv2.VideoCapture(0).release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main_video()

