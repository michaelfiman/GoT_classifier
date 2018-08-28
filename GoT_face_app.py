'''
    Author: 
        Michaelfi
        
    Date: 
        4.7.18
    
    Description: 
        App using a pretrained CNN model to recognize GoT charachters in a picture
    
    Python Version:
        3.5
'''

import argparse
import sys
import tensorflow as tf
import random
import cv2 as cv
from PIL import Image
import time
import numpy as np
from utils import load_specific_image, get_json_data, num_to_name

def pic_to_predict(pic, predict_fn):
    X = pic.reshape(96 * 96 * 3).flatten().tolist()
    model_input = tf.train.Example(features= tf.train.Features(feature={'x': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=X))}))
    model_input = model_input.SerializeToString()
    output_dict = predict_fn({"inputs":[model_input]})    
    return num_to_name(int(output_dict["classes"][0]))

def run_test(predict_fn):
    
    print("\n\n\nRunning basic test on evaluation dataset for 5 examples:\n\n")
    
    for j in range(5):
        i = random.randint(1200, 1300)
        X = load_specific_image(data, i)
        prediction = pic_to_predict(X, predict_fn)
        print("This is {}".format(prediction))
        print("this should be {}".format(data[i]['annotation']['label'][0]))
        if (prediction == (data[i]['annotation']['label'][0])):
            print("SUCCESS\n")
        else:
            print("FAILURE\n")
        
       
def run_predict_image(image_file, predict_fn):
    face_cascade = cv.CascadeClassifier('/home/shared/anaconda3/pkgs/opencv-3.3.1-py36h0a11808_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    img = cv.imread(image_file)
    
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(30, 30)
    )
    print(len(faces))
    predictions = []
    for (x, y, w, h) in faces:        
        
        crop_img = img[y+40:y+h-20, x+10:x+w-10]
        crop_img = cv.resize(crop_img, dsize=(96, 96), interpolation=cv.INTER_NEAREST)
        cv.imwrite('crop_img.jpg', crop_img)
        prediction = pic_to_predict(crop_img, predict_fn)
        print(prediction)
        predictions.append(prediction)
        
    for i, (x, y, w, h) in enumerate(faces):
        cv.putText(img, predictions[i], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv.imwrite('img_with_boxes.jpg', img)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="path to image from GoT", type=str)
    parser.add_argument("-t", "--test", help="run test on 5 sample images", type=bool ,default=False)
    args = parser.parse_args()
    
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    data = get_json_data('GoT_Face_Labelling_Ep7.json')
    
    predict_fn = tf.contrib.predictor.from_saved_model("trained/trained_0/1531036933/")
    if args.test:
        run_test(predict_fn)
    else:
        run_predict_image(image_file=args.image_path, predict_fn=predict_fn)
    
                                            
    