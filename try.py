import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
import os


def dsc(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dsc(y_true, y_pred)

def IOU(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    thresh = 0.5
    y_true = K.cast(K.greater_equal(y_true, thresh), 'float32')
    y_pred = K.cast(K.greater_equal(y_pred, thresh), 'float32')
    union = K.sum(K.maximum(y_true, y_pred)) + K.epsilon()
    intersection = K.sum(K.minimum(y_true, y_pred)) + K.epsilon()
    iou = intersection/union
    return iou

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = load_model('out.h5', custom_objects={'dice_loss': dice_loss, 'IOU': IOU, 'dsc': dsc, 'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m})




        
image_path = os.path.join("uploads", "checks3.jpg")

image = cv2.imread(image_path)
small_img = cv2.resize(image, (512, 512))
small_img = np.array(small_img)
small_img = small_img[None,:,:,:]
prediction = model.predict(small_img)[0] * 255
crack_image = cv2.resize(prediction, (512,512))

output_path = 'outputResult.png'
cv2.imwrite(output_path, crack_image)

b, g, r = cv2.split(crack_image)
z = np.zeros_like(g)
crack_image = cv2.merge((z, z, r))
crack_image = crack_image.astype(np.uint8)
image2=image
image2=cv2.resize(image2, (512,512))
result = cv2.addWeighted(image2, 1, crack_image, 1,0)
result = result.astype(np.uint8) 
           

output_path = 'outputOverlapped.png'
cv2.imwrite(output_path, result)

# print(predictions)

   



