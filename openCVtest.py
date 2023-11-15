import cv2 as cv


img = cv.imread('C:/Users/emilb/NTNU/Master/bilder/emil_kontakt_oss.jpg', -1)
img2 = cv.imread('C:/Users/emilb/NTNU/Master/bilder/iv_dagene_bilde.jpg', -1)
#C:\\Documents\\example.tx



# Endrer storleiken på bildet
img = cv.resize(img, (400,500))
img2 = cv.resize(img2, (0,0), fx= 0.3, fy=0.3)

height = img2.shape[0]
width = img2.shape[1]
print("Width and height: " + str(width) + " " +str(height))


#Linje gjennom bildet
#img = cv.line(img, (0,0), (width, height), (0, 255, 0), 1)
img = cv.line(img, (100,0), (100, width), (0, 255, 0), 5)

# Farge greier
hsv = cv. cvtColor(img2, cv.COLOR_BGR2HSV)


cv.imshow("Display window", img)
#cv.imshow("Display window", hsv)
k = cv.waitKey(0) # Wait for a keystroke in the windows
cv.destroyAllWindows()
