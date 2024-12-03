import cv2 as cv

#dosya oku
img=cv.imread(r'data/kizil_sac3.jpeg')
img_fer=cv.imread(r'data/kirmizi_ferrari.jpg')
b,g,r=cv.split(img)
bf,gf,rf=cv.split(img_fer)

#mavi saç
img_mavi=cv.merge((r,g,b))

#mavi ferrari
mv_fer=cv.merge((rf,gf,bf))

#sari saç
sari_sac=cv.merge((b,r,r))

#sari2
sari_sac=cv.merge((g,r,r))

#fotoyu göster
cv.imshow('Original',img)
cv.imshow('Mavi Sac',img_mavi)
cv.imshow('Sari Sac',sari_sac)
cv.imshow('Sari Sac 2',sari_sac)

cv.imshow('Kirmizi Ferrari',img_fer)
cv.imshow('Mavi Ferrari',mv_fer)
cv.waitKey(0)
cv.destroyAllWindows()

