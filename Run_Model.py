import cv2 as cv
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import join

data_dir = "./Data/BIPED/BIPED/edges"


def predict(prediction_model):
    image = cv.imread("C:\\Users\\saipa\\Desktop\\Dexined\\Dexined\\"
                      "Data\\BIPED\\BIPED\\edges\\imgs\\train\\rgbr\\real\\RGB_001.jpg")
    image = np.expand_dims(image, axis=0)
    pred = prediction_model.predict(image)
    pred = tf.nn.sigmoid(pred)
    pred = np.array(pred)
    cv.imwrite("C:\\Users\\saipa\\Desktop\\Pred.png", pred[0]*255)


def get_train_test_data():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    train_data_dir = join(data_dir, 'imgs', 'train', 'rgbr', 'real')
    train_label_dir = join(data_dir, 'edge_maps', 'train', 'rgbr', 'real')
    test_data_dir = join(data_dir, 'imgs', 'test', 'rgbr')
    test_label_dir = join(data_dir, 'edge_maps', 'test', 'rgbr')

    for each_train_data_file in listdir(train_data_dir):
        train_data = cv.imread(join(train_data_dir, each_train_data_file))
        x_train.append(train_data)
        each_train_label_file = join(train_label_dir, (each_train_data_file[:-4] + '.png'))
        train_label = cv.imread(each_train_label_file, cv.IMREAD_GRAYSCALE)
        train_label = ~train_label
        y_train.append(train_label)

    for each_test_data_file in listdir(test_data_dir):
        test_data = cv.imread(join(test_data_dir, each_test_data_file))
        x_test.append(test_data)
        each_test_label_file = join(test_label_dir, (each_test_data_file[:-4] + '.png'))
        test_label = cv.imread(each_test_label_file, cv.IMREAD_GRAYSCALE)
        test_label = ~test_label
        y_test.append(test_label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def compile_and_fit_model(model, x_train, y_train, x_test, y_test):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test))
