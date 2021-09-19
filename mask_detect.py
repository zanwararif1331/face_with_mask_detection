import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.applications import DenseNet121
import os
import cv2
from numpy import asarray
from PIL import Image
from functools import partial
from keras.applications import imagenet_utils

### Organizing classes into training, validation, and test
print("Organizing datasets")
masked_imgs = os.listdir('datasets/compiled/with_mask_copy')
no_mask_imgs = os.listdir('datasets/compiled/without_mask_copy')

i = 0
for img in masked_imgs:
    if i < int(0.64*len(masked_imgs)):
        os.rename('datasets/compiled/with_mask_copy/' + img, 'datasets/compiled/train/mask/' + img)
        i += 1
    elif i < int(0.80*len(masked_imgs)):
        os.rename('datasets/compiled/with_mask_copy/' + img, 'datasets/compiled/validation/mask/' + img)
        i += 1
    else:
        os.rename('datasets/compiled/with_mask_copy/' + img, 'datasets/compiled/test/mask/' + img)
        i += 1

j = 0
for img in no_mask_imgs:
    if j < int(0.64*len(no_mask_imgs)):
        os.rename('datasets/compiled/without_mask_copy/' + img,'datasets/compiled/train/no-mask/' + img)
        j += 1
    elif j < int(0.80*len(no_mask_imgs)):
        os.rename('datasets/compiled/without_mask_copy/' + img, 'datasets/compiled/validation/no-mask/' + img)
        j += 1
    else:
        os.rename('datasets/compiled/without_mask_copy/' + img, 'datasets/compiled/test/no-mask/' + img)
        j += 1

print("Images reorganized")

epochs = 2
batch_size = 16
train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True,
    rescale=1./255)

validation_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True,
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        batch_size=batch_size,
		directory='datasets/compiled/train/',
        target_size=(224, 224), 
        classes = ['no-mask','mask'],
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        batch_size=batch_size,
        directory='datasets/compiled/validation/',
        target_size=(224, 224), 
        classes = ['no-mask','mask'],
        class_mode='categorical')

### Pre-trained Model (DenseNet121 trained on Imagenet)
model = tf.keras.applications.DenseNet121(include_top=False,weights='imagenet',input_shape=(224,224,3),classes=2)


# Transfer Learning
for i in model.layers:
    i.trainable = False

global_avg = tf.keras.layers.GlobalAveragePooling2D()(model.output)
flatten = tf.keras.layers.Flatten()(global_avg)
# drop_out = tf.keras.layers.Dropout(0.4)(flatten)
out = tf.keras.layers.Dense(2,activation='softmax')(flatten)
densenet = tf.keras.Model(inputs=[model.input],outputs=[out])
densenet.summary()

densenet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss="binary_crossentropy",metrics=["accuracy"])

history = densenet.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

densenet.save('densenet121_detection_model.h5')

### Metrics
# Testing directories
test_mask_imgs = os.listdir('datasets/compiled/test/mask')
test_no_mask_imgs = os.listdir('datasets/compiled/test/no-mask')

# Numbers of images
total_mask = len(test_mask_imgs)
total_no_mask = len(test_no_mask_imgs)
total = total_mask + total_no_mask

# Confusion Matrix variables
mask_correct = 0
mask_incorrect = 0
no_mask_correct = 0
no_mask_incorrect = 0

# Mask class
for img in test_mask_imgs:
    
    tmp = Image.open('datasets/compiled/test/mask/'+ img)
    test_img = np.asarray(tmp)
    test = test_img.copy()
    test.resize(1,224,224,3)
    ans = densenet.predict(test)
    print(ans)
    if ans[0][1] > ans[0][0]:
        mask_correct += 1
        print("mask")
    elif ans[0][1] < ans[0][0]:
        mask_incorrect += 1
        print("no-mask")
        
    # Mask recall
    mask_recall = mask_correct/total_mask

# No-mask class
for img in test_no_mask_imgs:
    
    tmp = Image.open('datasets/compiled/test/no-mask/' + img)
    test_img = np.asarray(tmp)
    test = test_img.copy()
    test.resize(1,224,224,3)
    ans = densenet.predict(test)
    print(ans)
    
    if ans[0][0] > ans[0][1]:
        no_mask_correct += 1
        print("no-mask")
    elif ans[0][0] < ans[0][1]:
        no_mask_incorrect += 1
        print("mask")
        
    # No-mask recall
    no_mask_recall = no_mask_correct/total_no_mask

# Mask precision
mask_precision = (mask_correct)/(mask_correct + mask_incorrect)

# No-mask precision
no_mask_precision = (no_mask_correct)/(no_mask_correct + no_mask_incorrect)

# Mask F1 score
mask_f1 = 2 * ((mask_recall * mask_precision)/(mask_recall + mask_precision))

# No-mask F1 score
no_mask_f1 = 2 * ((no_mask_recall * no_mask_precision)/(no_mask_recall + no_mask_precision))

# Classification accuracy
accuracy = (mask_correct + no_mask_correct)/total

# Weighted recall
recall = ((mask_recall*total_mask)+(no_mask_recall*total_no_mask))/(total_mask+total_no_mask)

# Weight precision
precision = ((mask_precision*total_mask)+(no_mask_precision*total_no_mask))/(total_mask+total_no_mask)

# Weighted F1 score
f1 = 2 * ((recall * precision)/(recall + precision))

print("Mask correct predictions: {}".format(mask_correct))
print("Mask incorrect predictions: {}".format(mask_incorrect))

print("No-mask correct predictions: {}".format(no_mask_correct))
print("No-mask incorrect predictions: {}".format(no_mask_incorrect))

print("Mask total images: {}".format(total_mask))
print("No-mask total images: {}".format(total_no_mask))

print("Total test images: {}".format(total))

print("Mask recall: {}".format(mask_recall))
print("No-mask recall: {}".format(no_mask_recall))

print("Mask precision: {}".format(mask_precision))
print("No-mask precision: {}".format(no_mask_precision))

print("Mask F1 score: {}".format(mask_f1))
print("No-mask F1 score: {}".format(no_mask_f1))

print("Classification accuracy: {}".format(accuracy))
print("Weighted recall: {}".format(recall))
print("Weighted precision: {}".format(precision))
print("Weighted F1 score: {}".format(f1))

### Metrics
# Testing directories
test_mask_imgs = os.listdir('datasets/compiled/train/mask')
test_no_mask_imgs = os.listdir('datasets/compiled/train/no-mask')

# Numbers of images
total_mask = len(test_mask_imgs)
total_no_mask = len(test_no_mask_imgs)
total = total_mask + total_no_mask

# Confusion Matrix variables
mask_correct = 0
mask_incorrect = 0
no_mask_correct = 0
no_mask_incorrect = 0

# Mask class
for img in test_mask_imgs:
    
    tmp = Image.open('datasets/compiled/train/mask/'+ img)
    test_img = np.asarray(tmp)
    test = test_img.copy()
    test.resize(1,224,224,3)
    ans = densenet.predict(test)
    print(ans)
    if ans[0][1] > ans[0][0]:
        mask_correct += 1
        print("mask")
    elif ans[0][1] < ans[0][0]:
        mask_incorrect += 1
        print("no-mask")
        
    # Mask recall
    mask_recall = mask_correct/total_mask

# No-mask class
for img in test_no_mask_imgs:
    
    tmp = Image.open('datasets/compiled/train/no-mask/' + img)
    test_img = np.asarray(tmp)
    test = test_img.copy()
    test.resize(1,224,224,3)
    ans = densenet.predict(test)
    print(ans)
    
    if ans[0][0] > ans[0][1]:
        no_mask_correct += 1
        print("no-mask")
    elif ans[0][0] < ans[0][1]:
        no_mask_incorrect += 1
        print("mask")
        
    # No-mask recall
    no_mask_recall = no_mask_correct/total_no_mask

# Mask precision
mask_precision = (mask_correct)/(mask_correct + mask_incorrect)

# No-mask precision
no_mask_precision = (no_mask_correct)/(no_mask_correct + no_mask_incorrect)

# Mask F1 score
mask_f1 = 2 * ((mask_recall * mask_precision)/(mask_recall + mask_precision))

# No-mask F1 score
no_mask_f1 = 2 * ((no_mask_recall * no_mask_precision)/(no_mask_recall + no_mask_precision))

# Classification accuracy
accuracy = (mask_correct + no_mask_correct)/total

# Weighted recall
recall = ((mask_recall*total_mask)+(no_mask_recall*total_no_mask))/(total_mask+total_no_mask)

# Weight precision
precision = ((mask_precision*total_mask)+(no_mask_precision*total_no_mask))/(total_mask+total_no_mask)

# Weighted F1 score
f1 = 2 * ((recall * precision)/(recall + precision))

print("Mask correct predictions: {}".format(mask_correct))
print("Mask incorrect predictions: {}".format(mask_incorrect))

print("No-mask correct predictions: {}".format(no_mask_correct))
print("No-mask incorrect predictions: {}".format(no_mask_incorrect))

print("Mask total images: {}".format(total_mask))
print("No-mask total images: {}".format(total_no_mask))

print("Total test images: {}".format(total))

print("Mask recall: {}".format(mask_recall))
print("No-mask recall: {}".format(no_mask_recall))

print("Mask precision: {}".format(mask_precision))
print("No-mask precision: {}".format(no_mask_precision))

print("Mask F1 score: {}".format(mask_f1))
print("No-mask F1 score: {}".format(no_mask_f1))

print("Classification accuracy: {}".format(accuracy))
print("Weighted recall: {}".format(recall))
print("Weighted precision: {}".format(precision))
print("Weighted F1 score: {}".format(f1))

load_model = keras.models.load_model('densenet121_detection_model.h5')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
cv2.namedWindow("COVID Mask Detection Video Feed")
webcam = cv2.VideoCapture(0) 

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    rval, im = webcam.read()
    im=cv2.flip(im,1,1)
    
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
 
    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f] 
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        result=load_model.predict(reshaped)
        print(result)
        if result[0][0] > result[0][1]:
            percent = round(result[0][0]*100,2)
        else:
            percent = round(result[0][1]*100,2)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label] + " " + str(percent) + "%", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    if im is not None:   
        cv2.imshow('COVID Mask Detection Video Feed', im)
    key = cv2.waitKey(10)
    
    # Exit
    if key == 27: #The Esc key
        break
        
# Stop video
webcam.release()

# Close all windows
cv2.destroyAllWindows()
