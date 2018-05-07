import libjevois as jevois
import cv2
import numpy as np
#import rome
import process
import os.path

def color_to_rome_color(color):
    return {'O': 'orange', 'G': 'green'}.get(color[0], 'none')

class CubeVision:

    def __init__(self):
        pass

    def processNoUSB(self, inframe):
        self.process(inframe,None)

    def process(self, inframe, outframe):

        local = os.path.dirname(process.__file__)
        cam_file = os.path.join(local,"jevois_640.txt")
        cal_params = cv2.FileStorage(cam_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = cal_params.getNode("cameraMatrix").mat()
        self.dist_coeffs = cal_params.getNode("dist_coeffs").mat()


        im = inframe.getCvBGR()
        dbg = process.vision(im, self.camera_matrix, self.dist_coeffs)
        #params = dict(
        #    entry_color = color_to_rome_color(entry_color),
        #    cylinder_color = color_to_rome_color(cylinder_color),
        #    entry_height = int(entry_h),
        #    entry_area = int(entry_area),
        #    cylinder_area = int(cylinder_area),
        #)

        #frame = rome.Frame('jevois_tm_cylinder_cam', **params)
        #data = frame.data()
        #jevois.sendSerial(data)

        if outframe is not None:
            outframe.sendCvBGR(dbg)

