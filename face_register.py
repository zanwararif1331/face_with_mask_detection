import cv2, time

webcam = 1
cam = cv2.VideoCapture(webcam)
id = input('ID :')
name = input('Nama :')
a = 0

faceDetector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

while True :
    a = a+1
    retV, frame = cam.read()
    warna = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceDetector.detectMultiScale(warna, 1.3, 5)

    for (x, y, w, h) in faces :
        cv2.imwrite('DataSet/User.'+str(id)+'.'+str(name)+'.'+str(a)+'.jpg',warna[y:y+h,x:x+w])
        frame = cv2.rectangle(frame, (x, y), (x+w, y+w), (0,0,255), 5)
        rec_face = warna [y : y + w, x : x + w]
    cv2.imshow('WEBCAM', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27 :
        break
cam.release()
cv2.destroyAllWindows()