import cv2 as cv 

cap=cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    b, g, r = cv.split(frame)
    
    rgb_frame = cv.merge((r, g, b))
    cv.imshow('RGB  kanal', rgb_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break   
    cap.release()    
    cv.destroyAllWindows()