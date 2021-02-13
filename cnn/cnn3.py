import numpy as np 

import keras
from keras import layers
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D ,AveragePooling2D
from keras.layers import Activation , concatenate
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from more_itertools import locate
from keras.utils import to_categorical
from sklearn.base import BaseEstimator
from keras.utils import to_categorical

class CNN3(BaseEstimator):
    
    def __init__(self):
        pass


    def fit(self, X, Y,batch_size=100,epochs=50):
        X = X.reshape(-1, 20,300, 1)
        X = X.astype('float32')
        self.Y_dic = {
                    "feet": 0,
                    "left_hand": 1,
                    "right_hand": 2,
                    "tongue": 3
                }
        
        classes = np.unique(Y)
        numeric_Y = np.zeros(Y.shape,dtype=int)
        for classe in classes:
            a = (Y == classe)
            numeric_Y[a] = int(self.Y_dic[classe])
            
        # Change the labels from categorical to one-hot encoding
        Y_one_hot = to_categorical(numeric_Y)
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_one_hot, test_size=0.2)
        
        #-----------------------------------------------------------------------------------------------
        
        # Architecture of the Model
        
        self.hfcnn_model = Sequential()
        self.hfcnn_model.add(Conv2D(1, kernel_size=(1, 25),strides=(1, 1),
                                   activation=('elu'),use_bias=True,input_shape=(20,300,1),padding='valid'))
        #hfcnn_model.add(BatchNormalization())
        #hfcnn_model.add(Activation('relu'))
        #self.hfcnn_model.add(AveragePooling2D(pool_size=(1, 3),padding='same'))
        self.hfcnn_model.add(Dropout(0.5))

        self.hfcnn_model.add(Conv2D(1, (20, 1), strides=(1, 1),activation=('elu'),use_bias=True,padding='valid'))
        #hfcnn_model.add(BatchNormalization())
        #hfcnn_model.add(Activation('relu'))
        self.hfcnn_model.add(AveragePooling2D(pool_size=(1, 3),padding='valid'))
        self.hfcnn_model.add(Dropout(0.5))
        
        self.hfcnn_model.add(Conv2D(1, kernel_size=(1, 6),strides=(1, 1),
                                   activation=('elu'),use_bias=True,padding='valid'))
        self.hfcnn_model.add(Dropout(0.5))
        
        self.hfcnn_model.add(Conv2D(1, kernel_size=(1, 4),strides=(1, 1),
                                   activation=('elu'),use_bias=True,padding='valid'))
        self.hfcnn_model.add(AveragePooling2D(pool_size=(1, 3),padding='valid'))
        self.hfcnn_model.add(Dropout(0.5))
        
        self.hfcnn_model.add(Conv2D(1, kernel_size=(1, 4),strides=(1, 1),
                                   activation=('elu'),use_bias=True,padding='valid'))
        self.hfcnn_model.add(AveragePooling2D(pool_size=(1, 3),padding='valid'))
        self.hfcnn_model.add(Dropout(0.5))
        self.hfcnn_model.add(Conv2D(1, kernel_size=(1, 4),strides=(1, 1),
                                   activation=('elu'),use_bias=True,padding='valid'))
        self.hfcnn_model.add(AveragePooling2D(pool_size=(1, 3),padding='valid'))
        #self.hfcnn_model.add(Dropout(0.5))
        
        self.hfcnn_model.add(Flatten())

        
        self.hfcnn_model.add(Dense(4,activation=('softmax'),use_bias=True))
        #hfcnn_model.add(BatchNormalization())
        #hfcnn_model.add(Activation('softmax'))
        
        self.hfcnn_model.compile(loss=keras.losses.categorical_crossentropy, 
                                optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        
        hfcnn_train_dropout = self.hfcnn_model.fit(X_train, Y_train, batch_size=batch_size, 
                                              epochs=epochs, verbose=1, validation_data=(X_valid, Y_valid))
        return self

    def predict(self, V):
        V = V.reshape(-1, 20,300, 1)
        predicted_classes = self.hfcnn_model.predict(V)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
        reversed_Y_dic = {value : key for (key, value) in self.Y_dic.items()}
        p_classes = np.unique(predicted_classes)
        string_predicted_classes = [None] * len(predicted_classes)

        for item in p_classes:
            a =  list(locate(predicted_classes, lambda x: x == item))
            for aa in a :
                string_predicted_classes[aa] = (reversed_Y_dic[item])
        
        return string_predicted_classes