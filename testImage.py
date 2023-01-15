import cv2
import numpy as np
from keras.models import load_model

model = load_model('model/model_file_30epoches.h5')
faceDetect= cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

labels_dict= {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral',5:'sad',6:'surprise'}


frame = cv2.imread("pexels-andrea-piacquadio-3812743.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#len(number_of_img),img_hieght,  img_width, channel
faces = faceDetect.detectMultiScale(gray, 1.3, 3)
for x,y,w,h in faces:
    sub_face_img = gray[y:y+h, x:x+w ]
    resized= cv2.resize(sub_face_img,(48,48))
    normalize = resized/255.0
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    result = model.predict(reshaped)
    # print('saaaaaaaaaaaaaaaaaaaaaaa',result)
    label = np.argmax(result, axis=1)[0]
    # print('fffffffffffffffffffffffffff',label)
    # print(x,y,w,h)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0 ,225),2)
    # cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50 ,225), 2)
    # cv2.rectangle(frame, (x,y-40), (x+w,h), (50, 50 ,225), 1)
    cv2.putText(frame, labels_dict[label], (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(225,255,255),2)
    
    
cv2.imshow("Frame", frame)
cv2.imwrite('img1.jpg',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
    