import cv2 
from matplotlib import pyplot as plt
import numpy as np
import sympy

class try_match:
    def __init__(self, frame):
        self.tb = cv2.imread('templates/fb.png', cv2.IMREAD_GRAYSCALE)
        self.frame = frame


    def matching_Compute(self):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.tb, None)
        kp2, des2 = orb.detectAndCompute(self.frame, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches=bf.match(des1,des2)
        matches=sorted(matches, key=lambda x:x.distance)
        matching_result = cv2.drawMatches(self.tb, kp1, self.frame, kp2, matches[:10], None)
        return matching_result
    