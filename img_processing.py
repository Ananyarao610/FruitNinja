import cv2
x=cv2.imread("smiley.png", 1)   #flag-can be color=1 or greyscale=0 or unchanged=-1
cv2.imshow("Img", x)
cv2.waitKey(0)

