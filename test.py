import cv2
import conf
import correction

image = cv2.imread(conf.IMAGE_PATH)

print("analyzing this image...")
best = correction.sfrs_calibrate(image)

print("new image saved in ", conf.OUTPUT_PATH)
cv2.imwrite(conf.OUTPUT_PATH, best)

