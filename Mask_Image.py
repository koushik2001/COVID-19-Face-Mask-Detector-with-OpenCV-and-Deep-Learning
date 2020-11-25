import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img , img_to_array
import numpy as np
import os
import matplotlib.pyplot as plt

prototxt = r'C:\Users\saiko\Desktop\deploy.prototxt'

weights_path = r'C:\Users\saiko\Desktop\SSD.caffemodel'

net = cv2.dnn.readNet(prototxt,weights_path)

deep_model =load_model(r'C:\Users\saiko\Desktop\new_improved_model.h5')

image = cv2.imread(r'C:\Users\saiko\Desktop\doublemask.jfif')

blob = cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))

#detecting faces
net.setInput(blob)
detections = net.forward()


(h,w) = image.shape[:2]


#look over the detections


for i in range(0,detections.shape[2]):

    confidence = detections[0,0,i,2]
    
    if confidence>0.5:
        # we need x,y coordinates
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY) = box.astype('int')

        # we need to ensure bounding boxes fall within the dimensions of the frame

        (startX,startY)=(max(0,startX),max(0,startY))
        (endX,endY)=(min(w-1,endX), min(h-1,endY))

        face=image[startY:endY, startX:endX]
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face=cv2.resize(face,(300,300))
        face=img_to_array(face)
        face=np.expand_dims(face,axis=0)


        prediction = deep_model.predict(face)

        if prediction==0:
            class_label = "Mask"
            color = (0,255,0)
            
        else:
            class_label = "No Mask"
            color = (0,0,255)
        

        #display the label and bounding boxes

        cv2.putText(image,class_label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(image,(startX,startY),(endX,endY),color,2)


cv2.imshow("OutPut",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
