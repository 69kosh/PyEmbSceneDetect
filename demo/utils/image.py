import numpy as np
import cv2


def generateImage(array:np.array, filename:str):
    a = array.copy()
    a = np.clip(a, 0.0, 1.0)*255
    a = np.expand_dims(a.astype(np.uint8), axis = 2)
    a = np.concatenate((a, a, a), axis = 2)
    cv2.imwrite(filename, a)


