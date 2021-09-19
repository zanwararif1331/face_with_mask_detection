import cv2, time

webcam = 1
cam = cv2.VideoCapture(webcam)
font = cv2.FONT_HERSHEY_SIMPLEX
bw_threshold = 100

mouth_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
face_mask = cv2.CascadeClassifier('haarcascade/cascade.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_righteye.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_lefteye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training.xml')

while True :
    retV, frame = cam.read()
    img = cv2.flip(frame,1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    #cv2.imshow('black_and_white', black_and_white)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img,'Tidak Ada Orang',(250,50),font,1,(255,0,0),2)
    else:
        for (x, y, w, h) in faces:
           
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
            
            nose_rects = nose_cascade.detectMultiScale(gray, 1.5, 5)
            
            mask_rects = face_mask.detectMultiScale(gray, 1.5, 5)

            reye_rects = reye_cascade.detectMultiScale(gray, 1.5, 5)

            leye_rects = leye_cascade.detectMultiScale(gray, 1.5, 5)
            
        if(len(mouth_rects) == 0 and len(nose_rects) == 0):
            cv2.putText(img,'Menggunakan Masker',(170,50),font,1,(0,255,0),2)
            cv2.putText(img,'Hidung Dan Mulut Tertutup',(150,450),font,1,(250,255,0),2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 5)
            
            for (mx, my, mw, mh) in mask_rects:
                if(y < my < y + h):
                    cv2.putText(img,'Masker',(mx,my),font,0.5,(250,255,0),2)
                    cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (250,255,0), 2)
                    break
        
        elif(len(mouth_rects) == 0 and len(nose_rects) != 0):
            cv2.putText(img,'Pengunaan Masker Salah',(150,50),font,1,(0, 255, 255),2)
            cv2.putText(img,'Hidung Tidak Tertutup',(150,450),font,1,(0, 0, 255),2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,255), 5)

            for (mx, my, mw, mh) in nose_rects:
                if(y < my < y + h):
                    cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                    break
            
        elif(len(mouth_rects) != 0 and len(nose_rects) == 0):
            cv2.putText(img,'Pengunaan Masker Salah',(150,50),font,1,(0, 251, 255),2)
            cv2.putText(img,'Mulut tidak Tertutup',(150,450),font,1,(0, 0, 255),2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,255), 5)
            
            for (mx, my, mw, mh) in mouth_rects:
                if(y < my < y + h):
                    cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                    break
            
        else:
            for (mx, my, mw, mh) in mouth_rects:
                if(y < my < y + h):
                    cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                    break
                    
            for (mx, my, mw, mh) in nose_rects:
                if(y < my < y + h):
                    cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
                    break
            id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if (id==1):
                id='Zanwar'
            cv2.putText(img, str(id),(x+40,y-10), font, 1, (0,0,255), 2)
            rec_face = gray[y : y + w, x : x + w]        
            cv2.putText(img,'Tidak Menggunakan Masker',(150,50),font,1,(0, 0, 255),2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 5)

    cv2.imshow('Deteksi Masker', img)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()