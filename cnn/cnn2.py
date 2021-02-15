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

class CNN2(BaseEstimator):
    
    def __init__(self):
        pass


    def fit(self, X, Y,batch_size=50,epochs=10):
        X = X.reshape((-1, 20, 28, 28,1))
     #    X = X.astype('float32')
     #    self.Y_dic = {
     #          0: 0,
     #          1: 1,
     #        }

     #    numeric_label = np.zeros(Y.shape,dtype=int)
     #    classes = np.unique(Y)
     #    for classe in classes:
     #        a = (labels == classe)
          #   numeric_label[a] = int(self.Y_dic[classe])
            
        # Change the labels from categorical to one-hot encoding
        labels_one_hot = to_categorical(Y)
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, labels_one_hot, test_size=0.2)
        
        #-----------------------------------------------------------------------------------------------
        
        # Architecture of the Model
        
        inputA = Input(shape=(28,28,1))
        inputB = Input(shape=(28,28,1))
        inputC = Input(shape=(28,28,1))
        inputD = Input(shape=(28,28,1))
        inputE = Input(shape=(28,28,1))
        inputF = Input(shape=(28,28,1))
        inputG = Input(shape=(28,28,1))
        inputH = Input(shape=(28,28,1))
        inputI = Input(shape=(28,28,1))
        inputJ = Input(shape=(28,28,1))
        inputK = Input(shape=(28,28,1))
        inputL = Input(shape=(28,28,1))
        inputM = Input(shape=(28,28,1))
        inputN = Input(shape=(28,28,1))
        inputO = Input(shape=(28,28,1))
        inputP = Input(shape=(28,28,1))
        inputQ = Input(shape=(28,28,1))
        inputR = Input(shape=(28,28,1))
        inputS = Input(shape=(28,28,1))
        inputT = Input(shape=(28,28,1))

        A = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputA)
        A = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(A)
        A = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(A)
        #A = Dense(256,activation=('relu'),use_bias=True)(A)
        A= layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(A)
        A = Flatten()(A)



        B = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputB)
        B = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(B)
        B = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(B)
        #B = Dense(256,activation=('relu'),use_bias=True)(B)
        B = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(B)
        B = Flatten()(B)


        C = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputC)
        C = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(C)
        C = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(C)
        #C = Dense(256,activation=('relu'),use_bias=True)(C)
        C = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(C)
        C = Flatten()(C)



        D = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputD)
        D = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(D)
        D = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(D)
        #D = Dense(256,activation=('relu'),use_bias=True)(D)
        D = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(D)
        D = Flatten()(D)


        E = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputE)
        E = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(E)
        E = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(E)
        #E = Dense(256,activation=('relu'),use_bias=True)(E)
        E = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(E)
        E = Flatten()(E)


        F = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputF)
        F = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(F)
        F = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(F)
        #F = Dense(256,activation=('relu'),use_bias=True)(F)
        F = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(F)
        F = Flatten()(F)



        G = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputG)
        G = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(G)
        G = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(G)
        #G = Dense(256,activation=('relu'),use_bias=True)(G)
        G = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(G)
        G = Flatten()(G)



        H = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputH)
        H = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(H)
        H = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(H)
        #H = Dense(256,activation=('relu'),use_bias=True)(H)
        H = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(H)
        H = Flatten()(H)



        I = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputI)
        I = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(I)
        I = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(I)
        #I = Dense(256,activation=('relu'),use_bias=True)(I)
        I = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(I)
        I = Flatten()(I)



        J = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputJ)
        J = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(J)
        J = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(J)
        #J = Dense(256,activation=('relu'),use_bias=True)(J)
        J = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(J)
        J = Flatten()(J)



        K = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputK)
        K = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(K)
        K = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(K)
        #K = Dense(256,activation=('relu'),use_bias=True)(K)
        K = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(K)
        K = Flatten()(K)



        L = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputL)
        L = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(L)
        L = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(L)
        #L = Dense(256,activation=('relu'),use_bias=True)(L)
        L = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(L)
        L = Flatten()(L)



        M = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputM)
        M = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(M)
        M = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(M)
        #M = Dense(256,activation=('relu'),use_bias=True)(M)
        M = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(M)
        M = Flatten()(M)



        N = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputN)
        N = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(N)
        N = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(N)
        #N = Dense(256,activation=('relu'),use_bias=True)(N)
        N = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(N)
        N = Flatten()(N)



        O = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputO)
        O = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(O)
        O = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(O)
        #O = Dense(256,activation=('relu'),use_bias=True)(O)
        O = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(O)
        O = Flatten()(O)



        P = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputP)
        P = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(P)
        P = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(P)
        #P = Dense(256,activation=('relu'),use_bias=True)(P)
        P = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(P)
        P = Flatten()(P)



        Q = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputQ)
        Q = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(Q)
        Q = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(Q)
        #Q = Dense(256,activation=('relu'),use_bias=True)(Q)
        Q = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(Q)
        Q = Flatten()(Q)



        R = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputR)
        R = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(R)
        R = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(R)
        #R = Dense(256,activation=('relu'),use_bias=True)(R)
        R = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(R)
        R = Flatten()(R)



        S = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputS)
        S = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(S)
        S = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(S)
        #S = Dense(256,activation=('relu'),use_bias=True)(S)
        S = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(S)
        S = Flatten()(S)



        T = Conv2D(10, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(inputT)
        T = Conv2D(14, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(T)
        T = Conv2D(18, kernel_size=(3, 3),padding='same',activation=('relu'),use_bias=True)(T)
        #T = Dense(256,activation=('relu'),use_bias=True)(T)
        T = layers.AveragePooling2D(
             pool_size=(2, 2), strides=(3,3), padding="valid", data_format="channels_last")(T)
        T = Flatten()(T)


        merge_Model = concatenate([A, B, C, D, E, F, G, H , I, J, K, L, M, N, O, P, Q, R, S, T] )


        FC = Dense(1024,activation=('relu'),use_bias=True)(merge_Model)

        SOFTMAX = Dense(4,activation=('softmax'),use_bias=True)(FC)

        self.model = Model(inputs=[inputA, inputB ,inputC, inputD,inputE,
                              inputF,inputG, inputH,inputI, inputJ,inputK,
                              inputL , inputM, inputN ,inputO, inputP , 
                             inputQ, inputR , inputS, inputT],
                      outputs=SOFTMAX)
        
        self.model.compile(loss=keras.losses.categorical_crossentropy, 
                                optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        
        
        input1 = X_train[:,0,:,:,:]
        input2 = X_train[:,1,:,:,:]
        input3 = X_train[:,2,:,:,:]
        input4 = X_train[:,3,:,:,:]
        input5 = X_train[:,4,:,:,:]
        input6 = X_train[:,5,:,:,:]
        input7 = X_train[:,6,:,:,:]
        input8 = X_train[:,7,:,:,:]
        input9 = X_train[:,8,:,:,:]
        input10 = X_train[:,9,:,:,:]
        input11 = X_train[:,10,:,:,:]
        input12 = X_train[:,11,:,:,:]
        input13 = X_train[:,12,:,:,:]
        input14 = X_train[:,13,:,:,:]
        input15 = X_train[:,14,:,:,:]
        input16 = X_train[:,15,:,:,:]
        input17 = X_train[:,16,:,:,:]
        input18 = X_train[:,17,:,:,:]
        input19 = X_train[:,18,:,:,:]
        input20 = X_train[:,19,:,:,:]

        valid_input1 = X_valid[:,0,:,:,:]
        valid_input2 = X_valid[:,1,:,:,:]
        valid_input3 = X_valid[:,2,:,:,:]
        valid_input4 = X_valid[:,3,:,:,:]
        valid_input5 = X_valid[:,4,:,:,:]
        valid_input6 = X_valid[:,5,:,:,:]
        valid_input7 = X_valid[:,6,:,:,:]
        valid_input8 = X_valid[:,7,:,:,:]
        valid_input9 = X_valid[:,8,:,:,:]
        valid_input10 = X_valid[:,9,:,:,:]
        valid_input11 = X_valid[:,10,:,:,:]
        valid_input12 = X_valid[:,11,:,:,:]
        valid_input13 = X_valid[:,12,:,:,:]
        valid_input14 = X_valid[:,13,:,:,:]
        valid_input15 = X_valid[:,14,:,:,:]
        valid_input16 = X_valid[:,15,:,:,:]
        valid_input17 = X_valid[:,16,:,:,:]
        valid_input18 = X_valid[:,17,:,:,:]
        valid_input19 = X_valid[:,18,:,:,:]
        valid_input20 = X_valid[:,19,:,:,:]
        
        
        cwcnn_train_dropout = self.model.fit([input1 ,input2 ,input3 ,input4 ,input5 ,input6 ,input7 ,input8 ,input9 ,input10 ,
                                input11 ,input12 ,input13 ,input14 , input15 ,input16 ,input17 , input18 ,input19 ,
                                input20], 
                                Y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=([valid_input1 ,valid_input2 ,valid_input3 ,valid_input4
                                                        ,valid_input5 ,valid_input6 ,valid_input7 ,valid_input8 ,valid_input9 
                                                        ,valid_input10 , valid_input11 ,valid_input12 ,
                                                        valid_input13 ,valid_input14 , valid_input15 ,
                                                        valid_input16 ,valid_input17 , valid_input18 ,
                                                        valid_input19 , valid_input20] 
                                ,Y_valid))
        return self

    def predict(self, V):
        V = V.reshape(-1, 20, 28, 28,1)
        
        input1 = V[:,0,:,:,:]
        input2 = V[:,1,:,:,:]
        input3 = V[:,2,:,:,:]
        input4 = V[:,3,:,:,:]
        input5 = V[:,4,:,:,:]
        input6 = V[:,5,:,:,:]
        input7 = V[:,6,:,:,:]
        input8 = V[:,7,:,:,:]
        input9 = V[:,8,:,:,:]
        input10 = V[:,9,:,:,:]
        input11 = V[:,10,:,:,:]
        input12 = V[:,11,:,:,:]
        input13 = V[:,12,:,:,:]
        input14 = V[:,13,:,:,:]
        input15 = V[:,14,:,:,:]
        input16 = V[:,15,:,:,:]
        input17 = V[:,16,:,:,:]
        input18 = V[:,17,:,:,:]
        input19 = V[:,18,:,:,:]
        input20 = V[:,19,:,:,:]

        predicted_classes = self.model.predict([input1 ,input2 ,input3 ,input4 ,input5 ,input6 ,input7 ,input8 ,input9 ,input10 ,
                                input11 ,input12 ,input13 ,input14 , input15 ,input16 ,input17 , input18 ,input19 ,
                                input20])
        predicted_classes = np.argmax(predicted_classes,axis=1)
     #    reversed_Y_dic = {value : key for (key, value) in self.Y_dic.items()}
     #    p_classes = np.unique(predicted_classes)
     #    string_predicted_classes = [None] * len(predicted_classes)

     #    for item in p_classes:
     #        a =  list(locate(predicted_classes, lambda x: x == item))
     #        for aa in a :
     #            string_predicted_classes[aa] = (reversed_Y_dic[item])
        
        return predicted_classes





