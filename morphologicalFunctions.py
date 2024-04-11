#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def applyThreshold(image, minV, maxV):
    _, thresh_img = cv2.threshold(image, thresh = minV, maxval = maxV, type = cv2.THRESH_BINARY)
    return thresh_img

def applyErode(image, kernel, iterationNum):
    eroded = cv2.erode(image, kernel, iterations = iterationNum)
    return eroded

def applyDilate(image, kernel, iterationNum):
    dilated = cv2.dilate(image, kernel, iterations = iterationNum)
    return dilated

def applyClosing(image, kernel):
    closed = cv2.morphologyEx(image.astype(np.float32), cv2.MORPH_CLOSE, kernel)
    return closed

def applyOpening(image, kernel):
    opened = cv2.morphologyEx(image.astype(np.float32), cv2.MORPH_OPEN, kernel)
    return opened