import cv2
import numpy as np

img = np.zeros((256,256),dtype=np.uint8)
for i in range(256):
    for j in range(128,256):
        img[i,j]=255

cv2.imshow('img',img)
cv2.imwrite('./examples/masks/008.jpg',img)
cv2.waitKey(0)
