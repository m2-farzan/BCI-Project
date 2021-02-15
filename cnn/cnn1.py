import numpy as np 

from sklearn.base import BaseEstimator

from keras.utils import to_categorical

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split

from more_itertools import locate

class CNN1(BaseEstimator):
    
    def __init__(self):
        pass


    def fit(self, X, Y,batch_size=64,epochs=50):
        # print(list(Y))
        X = X.reshape(-1, 32,40, 1)
            
        # Change the labels from categorical to one-hot encoding
        Y_one_hot = to_categorical(Y)
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_one_hot, test_size=0.2)
        
        # Architecture of the Model
        
        self.cwcnn_model = Sequential()
        self.cwcnn_model.add(Conv2D(64, kernel_size=(1, 20),strides=(1, 2),
                                   activation=('relu'),use_bias=True,input_shape=(32,40,1),padding='same'))
        #cwcnn_model.add(BatchNormalization())
        #cwcnn_model.add(Activation('relu'))
        #cwcnn_model.add(MaxPooling2D((2, 2),padding='same'))
        self.cwcnn_model.add(Dropout(0.5))

        self.cwcnn_model.add(Conv2D(64, (1, 5), strides=(1, 2),activation=('relu'),use_bias=True,padding='same'))
        #cwcnn_model.add(BatchNormalization())
        #cwcnn_model.add(Activation('relu'))
        #cwcnn_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.cwcnn_model.add(Dropout(0.5))

        self.cwcnn_model.add(Flatten())

        self.cwcnn_model.add(Dense(512 ,activation=('relu'),use_bias=True)) 
        #cwcnn_model.add(BatchNormalization())
        #cwcnn_model.add(Activation('relu'))
        self.cwcnn_model.add(Dropout(0.5))

        self.cwcnn_model.add(Dense(4,activation=('softmax'),use_bias=True))
        #cwcnn_model.add(BatchNormalization())
        #cwcnn_model.add(Activation('softmax'))
        
        self.cwcnn_model.compile(loss=keras.losses.categorical_crossentropy, 
                                optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        
        cwcnn_train_dropout = self.cwcnn_model.fit(X_train, Y_train, batch_size=batch_size, 
                                              epochs=epochs, verbose=1, validation_data=(X_valid, Y_valid))
        return self

    def predict(self, V):
        V = V.reshape(-1, 32,40, 1)
        predicted_classes = self.cwcnn_model.predict(V)
        predicted_classes = np.argmax(predicted_classes, axis=1)
        return predicted_classes

