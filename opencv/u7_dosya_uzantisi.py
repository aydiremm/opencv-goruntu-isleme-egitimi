import cv2 as cv 


img=cv.imread('data/satranc3x3.jpg')

b,g,r=cv.split(img)

mavi_img=cv.merge((r,g,b))

cv.imshow('Orijinal',img)
cv.imshow('Mavi dama',mavi_img)

#kaydetme
cv.imwrite(r'data/mavi_dama.jpg',mavi_img)

#tek kanal olarak göster
cv.imshow('Blue Kanalı',b)
cv.imshow('Green Kanalı',g)
cv.imshow('Red Kanalı',r)    


cv.waitKey(0)
cv.destroyAllWindows()
