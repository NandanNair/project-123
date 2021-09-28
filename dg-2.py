import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 
import os,ssl,time
if(not os.environ.get("PYTHONHTTPSVERIFY","")and
    getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context=ssl._create_unverified_context
x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
nclasses=len(classes)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scale=x_train/255
x_test_scale=x_test/255
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(x_train_scale,y_train)

y_pre=clf.predict(x_test_scale)
accuracy=accuracy_score(y_test,y_pre)
print(accuracy)


cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        u_l=(int(width/2-56),int(height/2-56))
        b_r=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,u_l,b_r,(0,255,0),2)
        roi=gray[u_l[1]:b_r[1],u_l[0]:b_r[0]]
        im_pil=Image.fromarray(roi)
        image_vW=im_pil.convert("L")
        image_vW_R=image_vW.resize((28,28),Image.ANTIALIAS)
        image_vW_R_inverted=PIL.ImageOps.invert(image_vW_R)
        pixel_filter=20
        minimum_pixel=np.percentile(image_vW_R_inverted,pixel_filter)
        image_vW_R_inverted_scale=np.clip(image_vW_R_inverted-minimum_pixel,0,255)
        max_pixel=np.max(image_vW_R_inverted)
        image_vW_R_inverted_scale=np.asarray(image_vW_R_inverted_scale)/max_pixel
        test_sample=np.array(image_vW_R_inverted_scale).reshape(1,784)
        test_pre=clf.predict(test_sample)
        print(test_pre)
        cv2.imshow("frame",gray)
        if cv2.waitkey(1)&0xFF == ord('q'):
            break
    except Exception as e:
            pass
cap.release()
cv2.destroyAllWindows()                   


