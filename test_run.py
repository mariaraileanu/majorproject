# Importing Libraries
import cv2
from cv2 import *
import os
import dlib
from scipy import interpolate
from pylab import *
from skimage import color
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from imutils import face_utils


from blush import blushing
from lipstick import apply_lipstick
import traceback




##################
# Facial Landmarks Detection
# _face_detector is used to detect faces

_face_detector = dlib.get_frontal_face_detector()

# _predictor is used to get facial landmarks of a given face
cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "model/trained_models/shape_predictor_68_face_landmarks.dat"))
_predictor = dlib.shape_predictor(model_path)

##################

# Constants
img_size = 128
classes = ['01 light', '02 medium', '03 dark']

# load json and create model
json_file = open('model/trained_models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/trained_models/model.h5")
print("Loaded model from disk")

# Testing the model

# test_im = cv2.imread('T001.jpg')


# test_im = cv2.resize(test_im, (img_size, img_size), cv2.INTER_LINEAR) / 255
# test_pred = np.argmax(loaded_model(test_im.reshape((1, img_size, img_size, 3))))
# text = classes[test_pred]
# print(text)


def ARmakeup(frame):
     ### AR MAKEUP ###

    global _face_detector
    # converting frame to grayscale to get landmarks
    test_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces to ignore backgroung and focus more on the person's face
    faces = _face_detector(test_img)
    landmarks = []
    text = ""
    try:
        intensity = 0.7 # default for light skin color
        # lipstick color for light skin color
        lip_color = (182., 120., 150.) #in BGR
        landmarks = _predictor(test_img, faces[0])

        # numpy array that contains the facial landmarks 
        shape = face_utils.shape_to_np(landmarks)
        d = faces[0]

        # cropping out the area of interest from the frame that is face and the surrounding
        test_im = cv2.resize(frame[d.top():d.bottom(),d.left():d.right()], (img_size, img_size), cv2.INTER_LINEAR) / 255
        test_pred = np.argmax(loaded_model(test_im.reshape((1, img_size, img_size, 3))))
        text = classes[test_pred]
        
        # Settings for medium skin tone
        if text == '02 medium':
            intensity = 0.6
            lip_color = (133., 51., 187.)
            
        # Settings for dark skin tone
        elif text == '03 dark':
            
            intensity = 0.3
            lip_color = (207., 40., 57.)  

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        # To crop and draw rectangle around face area for better and stable prediction (Uncomment the two lines below to draw rectangle on live cam)
        #inc = int(0.25*(d.bottom()-d.top()))
        #frame = cv2.rectangle(frame, (d.left(),d.top()-inc), (d.right(),d.bottom()+inc), (0,255,0), 2)
        
        # EyeLiner
        shape_list = shape.tolist()
        for i,j in enumerate(shape_list):
            shape_list[i] = (j[0], j[1])
        indices = [36,37,38,39,40,41,36]
        left_eye = [shape_list[i] for i in indices]
        indices = [42,43,44,45,46,47,42]
        right_eye = [shape_list[i] for i in indices]
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img , 'RGBA')
        draw.line(left_eye, fill=(0, 0, 0, 180), width=2)
        draw.line(right_eye, fill=(0, 0, 0, 180), width=2)
        frame = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
       
        # to define the size of blush dynamically for different sized images
        cheek_size = abs(shape[36][0]-shape[39][0])/2
        # landmarks for right cheek
        right_cheek_points = [
            (shape[14][1], shape[46][0]),
            (shape[14][1] - cheek_size, shape[46][0]),
            (shape[14][1] - cheek_size, shape[46][0] + cheek_size),
            (shape[14][1], shape[46][0] + cheek_size),
            (shape[14][1], shape[46][0])
            ]
        # left_cheek_points = [
        #     (shape[2][1], shape[19][0]),
        #     (shape[2][1] - cheek_size, shape[19][0]),
        #     (shape[2][1] - cheek_size, shape[19][0] - cheek_size),
        #     (shape[2][1], shape[19][0] - cheek_size),
        #     (shape[2][1], shape[19][0])
        #     ]
        # landmarks for lips, the complex implementation and checks are to cater possible errors due to camera lag 
        lips_points = np.array([[shape[48][1], shape[48][0]]])
        if shape[58][1] in lips_points[:,0]:
            lips_points = np.vstack([lips_points,[shape[58][1]+1, shape[58][0]]])
        else:
            lips_points = np.vstack([lips_points,[shape[58][1], shape[58][0]]])

        check = shape[57][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[57][0]]])

        check = shape[56][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[56][0]]])

        check = shape[54][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[54][0]]])

        check = shape[52][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[52][0]]])

        check = shape[50][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[50][0]]])

        check = shape[60][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[60][0]]])

        check = shape[67][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[67][0]]])

        check = shape[66][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[66][0]]])

        check = shape[65][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[65][0]]])

        check = shape[64][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[64][0]]])

        check = shape[63][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[63][0]]])

        check = shape[61][1]
        while (check  in lips_points[:,0]):
            check += 1
        lips_points = np.vstack([lips_points,[check, shape[61][0]]])
        lips_points = tuple(map(tuple, lips_points))


        # mid point of the face around which the right cheek points will be mirrored to apply blush on the left cheek as well 

        mid = shape[33][0]

        # Blush function that inputs the original frame, landmark points of right cheek, mid and intensity of blush
        frame = blushing(frame, np.array(right_cheek_points), mid, intensity)

        
        # Blush function that inputs the original frame, landmark points, and lip color
        frame = apply_lipstick(frame, np.array(lips_points), lip_color)
        return frame

    except Exception as e:
        print(traceback.format_exc())
        cv2.putText(frame, 'No Face Detected', (370, 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 2)

        return frame
    
    ### Landmarks Detection above ###


# AR makeup test on sample images from each class
test_im1 = cv2.imread('T004.jpg') # 4 1 13
test_im2 = cv2.imread('T002.jpg')
test_im3 = cv2.imread('T013.jpg')
test_out1 = ARmakeup(test_im1)
test_out2 = ARmakeup(test_im2)
test_out3 = ARmakeup(test_im3)
cv2.imwrite('test_out1.jpg', test_out1)
cv2.imwrite('test_out2.jpg', test_out2)
cv2.imwrite('test_out3.jpg', test_out3)


# Skin Tone Classification and AR makeup on live web cam
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    frame = ARmakeup(frame)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break




    