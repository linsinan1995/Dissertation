import numpy as np
import cv2
from PIL import Image
import os


def MatrixToImage(data):
    data = data
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def crop(filename,crop=False,show=False,output_img=False):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load(r'D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray) #,scaleFactor=1.5)
    #用于裁剪
    if crop:
        x = faces[0][0]
        y = faces[0][1]
        h = faces[0][2]
        w = faces[0][3]
        center = [(y + h / 2), (x + w / 2)]
        gray2 = gray[(center[0] - 100):(center[0] + 100), (center[1] - 100):(center[1] + 100)]
    else:
        for (x, y, h, w) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        gray2 = gray[(y + 2):(y + h - 1), (x + 2):(x + w - 1)]
    if show:
        cv2.namedWindow('faces Detected!')
        cv2.imshow('faces Detected!', gray)
        cv2.imwrite('faces.jpg', gray)
        cv2.waitKey(0)
        def MatrixToImage(data):
            data = data
            new_im = Image.fromarray(data.astype(np.uint8))
            return new_im
        MatrixToImage(gray2).show()

    return gray2







# 得到剪切后的矩阵
filenames = os.listdir(r'C:\Users\hasee\Desktop\IMM_face_db')
for j in range(40):
    for i in range(6):
        filename = 'C:\\Users\\hasee\\Desktop\\IMM_face_db\\' + filenames[6*j+i]
        crop_file = crop(filename,crop=True,show=False)
        if i == 0 and j == 0:
            x = crop_file.ravel()
        else:
            x = np.concatenate((x, crop_file.ravel()), axis=0)
#reshape
zz=x.reshape(40*6,200*200)


#检查剪切后的图片
for i in range(len(zz)):
    d = zz[i].reshape(200,200)
    filename2 = "C:\\Users\\hasee\\Desktop\\crop2\\" + "%d.jpg" %i
    cv2.imwrite(filename2, d,[int(cv2.IMWRITE_JPEG_QUALITY), 100])


#依次展示
for i in range(len(zz)):
    pic = zz[i,:]
    MatrixToImage(pic.reshape(200,200)).show()


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_cascade.load(r'D:\opencv\opencv\sources\data\haarcascades\haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load(r'D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

filename = r'C:\Users\hasee\Desktop\IMM_face_db\01-2m.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255, 251, 240),3)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255, 251, 240),3)

cv2.imwrite("C:\\Users\\hasee\\Desktop\\haar.jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.waitKey(0)



# 保存剪切后的图片
filenames = os.listdir(r'C:\Users\hasee\Desktop\IMM_face_db')
for filename in filenames:
    filename = 'C:\\Users\\hasee\\Desktop\\IMM_face_db\\' + filename
    img = cv2.imread(filename)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load(r'D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5)
    x = faces[0][0]
    y = faces[0][1]
    h = faces[0][2]
    w = faces[0][3]
    center = [(y + h/2),(x + w/2)]
    gray2 = gray[(center[0]-100):(center[0]+100), (center[1]-100):(center[1]+100)]
    MatrixToImage(gray2).show()


# test
if 3>2:
    filename = r'C:\Users\hasee\Desktop\IMM_face_db\05-3m.jpg'
    d=crop(filename,crop=True,show=False)



    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eye_cascade.load(r'D:\opencv\opencv\sources\data\haarcascades\haarcascade_eye.xml')

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load(r'D:\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        center = [(y + h / 2), (x + w / 2)]
        img2 = img[(center[0] - 100):(center[0] + 100), (center[1] - 100):(center[1] + 100)]
    cv2.imwrite("C:\\Users\\hasee\\Desktop\\haar.jpg",img2,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.waitKey(0)

filename = r'C:\Users\hasee\Desktop\IMM_face_db\05-1m.jpg'
d=crop(filename,False,True)