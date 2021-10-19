import cv2 as cv
import numpy as np
from os import listdir, mkdir
from os.path import join, exists
from utls.losses import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def predict(model_path, predict_data, results_dir):
    loaded_model = load_model(model_path)
    predictions = loaded_model(predict_data)
    np_predictions = np.array(predictions)

    if not exists(results_dir):
        mkdir(results_dir)

    total_files = np_predictions.shape[0]
    for index in range(total_files):
        print("Working on file {} of {}".format(index+1, total_files))
        pred = np.squeeze(np_predictions[index])
        file_name = 'pred_' + str(index+1)
        save_file = join(results_dir, file_name)
        cv.imwrite(save_file, pred*255.0)


def get_train_test_data(data_dir):
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
        y_train.append((train_label/255.0))

    for each_test_data_file in listdir(test_data_dir):
        test_data = cv.imread(join(test_data_dir, each_test_data_file))
        x_test.append(test_data)
        each_test_label_file = join(test_label_dir, (each_test_data_file[:-4] + '.png'))
        test_label = cv.imread(each_test_label_file, cv.IMREAD_GRAYSCALE)
        test_label = ~test_label
        y_test.append((test_label/255.0))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def compile_model(my_model):
    my_model.compile(loss={'output1': cross_entropy_balanced,
                           'output2': cross_entropy_balanced,
                           'output3': cross_entropy_balanced,
                           'output4': cross_entropy_balanced,
                           'output5': cross_entropy_balanced,
                           'output6': cross_entropy_balanced,
                           'fuse': cross_entropy_balanced,
                           },
                     metrics={'fuse': pixel_error},
                     optimizer='adam')

    return my_model


def train_model(my_model, x_train=None, y_train=None, x_test=None, y_test=None, batch_size=1, epochs=50,
                save_dir="./Trained_Model"):
    if not exists(save_dir):
        mkdir(save_dir)

    model_path = join(save_dir, 'Best_Model.h5')
    ckpt = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)

    my_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                 validation_data=(x_test, y_test), callbacks=[ckpt])
