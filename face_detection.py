import cv2
cap = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(gray,1.3,5)
    detections = cascade_classifier.detectMultiScale(gray,1.1,4)

    if (len(detections)>0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cap.destroyAllWindows()