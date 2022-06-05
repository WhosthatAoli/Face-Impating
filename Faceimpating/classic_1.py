import numpy as np
import cv2
img = cv2.imread('./test/image/basic.jpg')
print(img.shape)
mask = cv2.imread('./test/mask/mask.jpg',0)
print(mask.shape)
dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()