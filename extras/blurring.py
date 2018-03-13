import cv2
img = cv2.imread('rita.jpg')
blurimg = cv2.blur(img,(500,500))
cv2.imwrite('blurred.png',blurimg)