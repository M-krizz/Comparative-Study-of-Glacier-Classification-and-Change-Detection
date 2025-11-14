import cv2
img = cv2.imread("c:/Users/Administrator/Pictures/Video Projects/Spain-Soccer.jpg", 0)
print(img)
cv2.imshow('image',img)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
elif k==ord('s'):
    cv2.imwrite("football.jpg",img)
    cv2.destroyAllWindows()
    

