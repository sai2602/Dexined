import cv2 as cv
import numpy as np
import tensorflow as tf


def predict(prediction_model):
    image = cv.imread("C:\\Users\\SaiParimi\\Desktop\\complex_pred.jpg")
    image = np.expand_dims(image, axis=0)
    pred = prediction_model.predict(image)
    pred = tf.nn.sigmoid(pred)
    pred = np.array(pred)
    cv.imwrite("C:\\Users\\SaiParimi\\Desktop\\pred.png", pred[0]*255)
