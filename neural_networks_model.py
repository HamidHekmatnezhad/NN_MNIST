# region Imports and load data
import tensorflow as tf
import numpy as np
from random import randint
from matplotlib import pyplot as plt
import os 
import imageio
from scipy.special import expit, logit
import PIL
# endregion


class common:
    def np_data(self):
        self.x_train = np.load('data/mnist/x_train.npy')
        self.y_train = np.load('data/mnist/y_train.npy')
        self.x_test  = np.load('data/mnist/x_test.npy')
        self.y_test  = np.load('data/mnist/y_test.npy')

        assert self.x_train.shape == (60000, 28, 28)
        assert self.x_test.shape == (10000, 28, 28)
        assert self.y_train.shape == (60000,)
        assert self.y_test.shape == (10000,)

        # normalize 
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test  = self.x_test.astype('float32') / 255.0

    def keras_data(self):
            ''''''
            data = tf.keras.datasets.mnist         
            # load data
            (self.x_train, self.y_train), (self.x_test, self.y_test) = data.load_data()
            # normalize data
            self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
            # reshaping data for convert to image
            self.x_train = self.x_train.reshape(-1, 28, 28, 1)
            self.x_test = self.x_test.reshape(-1, 28, 28, 1)

    def predict_model(self, data):
        '''image must be 28x28'''

        result = self.model.predict(data)
        return result

    def predict_of_test_data(self, n=10):
        '''test model with test data, 
        retrun:
            - r is a string. T mean True and F mean False.
            - msgs is list of string. Correct or Wrong results.
            - ims_idx is list of index of images'''
        
        
        ims_idx = []
        for i in range(n):
            x = randint(0, 9999)
            ims_idx.append(x)

        results, msgs, r, msg = [], [], str(), ''

        for i in ims_idx:
            results.append(self.model.predict(self.x_test[i].reshape(1, 28, 28, 1)))

        for i in range(len(ims_idx)):
            if str(np.argmax(results[i])) == str(self.y_test[ims_idx[i]]):
                msg = "Correct " + str(np.argmax(results[i]))
                r += 'T '
            else:
                msg = "Wrong " + str(np.argmax(results[i])) + " - " + str(self.y_test[ims_idx[i]])
                r += 'F ' 

            msgs.append(msg)
            msg = ''
        
        return r, msgs, ims_idx
    
    def prepare_model(self, weights_path, show_info=False):
        ''''''

        # load weights
        try:
            self.model.load_weights(weights_path)

        except:
            raise "weights not found"

        info = self.model.evaluate(self.x_test, self.y_test)
        if show_info:
            return info

##########################################################################################################

class nn_MLP(common):
    def __init__(self, activation_function='sigmoid' ,optomizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        '''
        predict_of_test_data: Test Model with Test Data.'''
        
        self.np_data()
        self.activation_function = activation_function
        self.optomizer = optomizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self, model_name='4l', show_info=False):
        '''name of model: 4l, '''
        match model_name:

            case '4l':
                PATH = 'saved_weights/l4/main_4layers.weights.h5'
                self.model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(28, 28)),
                    tf.keras.layers.Dense(256, activation=self.activation_function),
                    tf.keras.layers.Dense(128, activation=self.activation_function),
                    tf.keras.layers.Dense(64, activation=self.activation_function),
                    tf.keras.layers.Dense(32, activation=self.activation_function),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
            
            case _:
                raise "model name not found"

        self.model.compile(optimizer=self.optomizer, loss=self.loss, metrics=self.metrics)

        if show_info:
            self.model.summary()
        self.prepare_model(PATH, show_info)
  
##########################################################################################################

class CNN(common):
    def __init__(self, activation_function_1='sigmoid', activation_function_2='relu', optomizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        ''''''

        self.keras_data()
        self.activation_function_1 = activation_function_1
        self.activation_function_2 = activation_function_2
        self.optomizer = optomizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self, model_name='smpl', show_info=False):
        '''model_name: smpl, 4l, dropout'''

        match model_name:

            case 'smpl':
                PATH = 'saved_weights/cnn_model/' + 'cnn_base.weights.h5'
                self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation_function_1, input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation_function_1),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=self.activation_function_2),
                tf.keras.layers.Dense(10, activation='softmax')
                ])
            
            case '4l':
                PATH = 'saved_weights/cnn_model/' + 'cnn_l4.weights.h5'
                self.model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation_function_1, input_shape=(28, 28, 1)),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation_function_1),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(256, activation=self.activation_function_2),
                    tf.keras.layers.Dense(128, activation=self.activation_function_2),
                    tf.keras.layers.Dense(64, activation=self.activation_function_2),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])

            case 'dropout':
                PATH = 'saved_weights/cnn_model/' + 'cnn_dropout.weights.h5'
                self.model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation_function_1, input_shape=(28, 28, 1)),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Conv2D(64, (3, 3), activation=self.activation_function_1),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(256, activation=self.activation_function_2),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(128, activation=self.activation_function_2),
                    tf.keras.layers.Dropout(0.2), 
                    tf.keras.layers.Dense(64, activation=self.activation_function_2),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])

            case _:
                raise "model name not found"

        self.model.compile(optimizer=self.optomizer, loss=self.loss, metrics=self.metrics)

        if show_info:
            self.model.summary()
        self.prepare_model(PATH, show_info)

##########################################################################################################

class RNN(common):
    def __init__(self, activation_function='relu', optomizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        ''''''

        self.np_data()
        self.activation_function = activation_function
        self.optomizer = optomizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self, model_name='smpl', show_info=False):
        '''model_name: smpl, 4l'''

        match model_name:
            case 'smpl':
                PATH = 'saved_weights/RNN_model/' + 'rnn_simple.weights.h5'
                self.model = tf.keras.Sequential([
                    tf.keras.layers.SimpleRNN(128, activation=self.activation_function, input_shape=(28, 28)),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])

            case '4l':
                PATH = 'saved_weights/RNN_model/' + 'rnn_simple2.weights.h5'
                self.model = model2 = tf.keras.Sequential([
                    tf.keras.layers.SimpleRNN(512, activation=self.activation_function, input_shape=(28, 28), return_sequences=True),
                    tf.keras.layers.SimpleRNN(256, activation=self.activation_function, return_sequences=True),
                    tf.keras.layers.SimpleRNN(128, activation=self.activation_function),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])

            case _:
                raise "model name not found"
            
        self.model.compile(optimizer=self.optomizer, loss=self.loss, metrics=self.metrics)

        if show_info:
            self.model.summary()
        self.prepare_model(PATH, show_info)

##########################################################################################################

class LSTM(common):
    def __init__(self, activation_function='relu', optomizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        ''''''

        self.np_data()
        self.activation_function = activation_function
        self.optomizer = optomizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self, model_name='smpl', show_info=False):
        '''model_name: smpl, 4l'''

        match model_name:
            case 'smpl':
                PATH = 'saved_weights/LSTM_model/' + 'lstm_1.weights.h5'
                self.model = tf.keras.models.Sequential([
                    tf.keras.layers.LSTM(128, input_shape=(28, 28), activation=self.activation_function),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])

            case _:
                raise "model name not found"

        self.model.compile(optimizer=self.optomizer, loss=self.loss, metrics=self.metrics)

        if show_info:
            self.model.summary()
        self.prepare_model(PATH, show_info)

##########################################################################################################

class GRU(common):
    def __init__(self, activation_function='tanh', optomizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        ''''''

        self.np_data()
        self.activation_function = activation_function
        self.optomizer = optomizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self, model_name='smpl', show_info=False):
        '''model_name: smpl, 4l'''

        match model_name:
            case 'smpl':
                PATH = 'saved_weights/gru_model/' + 'gru_model.weights.h5'
                self.model = tf.keras.models.Sequential([
                    tf.keras.layers.GRU(128, input_shape=(28, 28), activation=self.activation_function),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])

            case _:
                raise "model name not found"

        self.model.compile(optimizer=self.optomizer, loss=self.loss, metrics=self.metrics)

        if show_info:
            self.model.summary()
        self.prepare_model(PATH, show_info)

##########################################################################################################

class nn_basic(common):
    def __init__(self):

        prmtr = np.load('saved_weights/hyper_parameters.npy')
        
        self.input_nodes = 784
        self.hidden_nodes = int(prmtr[0])
        self.output_nodes = 10
        self.learning_rate = int(prmtr[1])

        self.activtion_function = lambda x : expit(x)

        self.inverse_activation_function = lambda x : logit(x)

        self.weigths_input_to_hidden_layer = np.random.default_rng().normal(0, pow(self.input_nodes, -0.5),
                                                          (self.hidden_nodes, self.input_nodes))
        self.weigths_hidden_to_output_layer = np.random.default_rng().normal(0, pow(self.hidden_nodes, -0.5),
                                                          (self.output_nodes, self.hidden_nodes))
        
        # load weights
        self.weigths_input_to_hidden_layer  = np.load('saved_weights/weights_input_to_hidden_layer.npy')
        self.weigths_hidden_to_output_layer = np.load('saved_weights/weights_hidden_to_output_layer.npy')

    def query(self, input_list):
        '''input_list: img 28x28'''
        inputs = np.array(input_list, ndmin=2).T

        x_hiddem = np.dot(self.weigths_input_to_hidden_layer, inputs)
        o_hidden = self.activtion_function(x_hiddem)

        x_output = np.dot(self.weigths_hidden_to_output_layer, o_hidden)
        o_output = self.activtion_function(x_output)

        return o_output

    def test_data(self, n=10):
        '''test model with test data, 
        retrun:
            - r is a string. T mean True and F mean False.
            - msgs is list of string. Correct or Wrong results.
            - ims_idx is list of index of images'''
        
        self.np_data()
        ims_idx = []
        for i in range(n):
            x = randint(0, 9999)
            ims_idx.append(x)

        results, msgs, r, msg = [], [], str(), ''

        for i in ims_idx:
            results.append(self.query(self.x_test[i].reshape(784)))

        for i in range(len(ims_idx)):
            if str(np.argmax(results[i])) == str(self.y_test[ims_idx[i]]):
                msg = "Correct " + str(np.argmax(results[i]))
                r += 'T '
            else:
                msg = "Wrong " + str(np.argmax(results[i])) + " - " + str(self.y_test[ims_idx[i]])
                r += 'F ' 

            msgs.append(msg)
            msg = ''
        
        return r, msgs, ims_idx

##########################################################################################################

class kernel_conv:
    def __init__(self):
        
        # define kernels for convolution
        self.knl_box_blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        self.knl_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.knl_edge_detect = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
        self.knl_edge_detect_2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.knl_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.knl_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        self.knl_gaussian_blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        self.knls = [self.knl_box_blur, self.knl_identity, self.knl_edge_detect, self.knl_edge_detect_2, self.knl_sharpen, self.knl_emboss, self.knl_gaussian_blur]

    def convolution(self, img, knl:list, dim='3d'):
        '''convolution of image with kernel
        img: image
        knl: kernel. list
        step: step size
        dim: dimension of image. 1d or 3d
        return: convoluted image'''

        # get image dimensions
        if dim == '3d':
            img_height, img_width, img_channels = img.shape
            im = np.zeros((img_height-2, img_width-2, img_channels))

            for c in range(3):
                i, j = 0, 0
                while j < img_width - 3:
                    while i < img_height - 3:
                        res = 0
                        for k in range(3):
                            for l in range(3):
                                res += img[i + k, j + l, c] * knl[k, l]
                        
                        if res < 0: res = 0
                        elif res > 255: res = 255
                        im[i, j, c] = res
                        i += 1
                    i = 0
                    j += 1
        
            return im

        elif dim == '1d':
            img_height, img_width = img.shape
            im = np.zeros((img_height-2, img_width-2))
            i, j = 0, 0
            while j < img_width - 3:
                while i < img_height - 3:
                    res = 0
                    for k in range(3):
                        for l in range(3):
                            res += img[i + k, j + l] * knl[k, l]
                    
                    if res < 0: res = 0
                    elif res > 255: res = 255
                    im[i, j] = res
                    i += 1
                i = 0
                j += 1

            return im
        
    def pooling(self, img, dim):
        if dim == '3d':
            img_height, img_width, img_channels = img.shape
            im = np.zeros((img_height // 2, img_width // 2, img_channels))
            i, j, c = 0, 0, 0
            while c < img_channels:
                while j < img_width - 2:
                    while i < img_height - 2:
                        im[i // 2, j // 2, c] = np.max(img[i:i + 2, j:j + 2, c]) # max pooling
                        i += 2
                    i = 0
                    j += 2
                i = 0
                j = 0
                c += 1

            return im

        elif dim == '2d':
            img_height, img_width = img.shape
            im = np.zeros((img_height // 2, img_width // 2))
            i, j = 0, 0
            while j < img_width - 2:
                while i < img_height - 2:
                    im[i // 2, j // 2] = np.max(img[i:i + 2, j:j + 2]) # max pooling
                    i += 2
                i = 0
                j += 2
            return im
        
##########################################################################################################    