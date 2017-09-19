#coding:utf-8

import numpy as numpy
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Input
from layers import ConvOffset2D

class myDNN(object):
    def __init__(self, patch_radius=4, channels=3):
        self.patch_radius = patch_radius
        self.channels = channels


    def preforward():
        # input
        inputs1 = l = Input((9, 9, 3), name='inputs1')

        # conv1
        l_offset = ConvOffset2D(3, name='conv1_offset')(l)
        l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv1', trainable=train)(l_offset)
        l = Activation('relu', name='conv1_relu')(l)
        l = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name='max_pool_1')(l)

        # conv2
        l_offset = ConvOffset2D(128, name='conv2_offset')(l)
        l = Conv2D(256, (3, 3), padding='same', name='conv2', trainable=train)(l_offset)
        l = Activation('relu', name='conv2_relu')(l)
        l = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name='max_pool_2')(l)

        l = Flatten()(l)
        l = Dense(256, name='fc1', trainable=train)(l)

        # ***************************************************
        # output
        # ***************************************************
        l = Dense(256, activation='sigmoid')(l)
        outputs = Dense(2, activation='softmax')(l)  #输出类型要改！！！
        model = Model(inputs=inputs1, outputs=outputs)
        optim = Adam()
        # optim = SGD(1e-3, momentum=0.99, nesterov=True)
        loss = categorical_crossentropy
        model.compile(optim, loss, metrics=['accuracy'])

        return model


    def forward(filename):
        '''
        fintue; include two branches.
        '''
        model = pretrain()
        model.load_weights(filename)

        # third branch -- encode the coorinate
        inputs2 = Input((2,), name='inputs2')
        m = Dense(256, name='fc_xc', trainable=train)(inputs2)


        # conactenate and output
        concatenated = concatenate([l, m])
        outputs = Dense(256, activation='sigmoid')(concatenated)
        outputs = Dense(2, activation='softmax')(outputs)  #输出类型要改！！！

        model = Model(inputs=[inputs1,inputs2], outputs=outputs)
        optim = Adam()
        # optim = SGD(1e-3, momentum=0.99, nesterov=True)
        loss = categorical_crossentropy
        model.compile(optim, loss, metrics=['accuracy'])


    def train(X, Xc, Y):
        # 1. pretrain
        t0 = time.time()
        Y_label = np_utils.to_categorical(y_train, 2)  # 这里的输出类别也要改！！！
        print("Pretrain...")
        pretrain(np.asarray(x_train), np.asarray(Y_label))
        print("Done...")
        print("Pretraining Time:", time.time() - t0)

        # 2. fintune
        print("Finetune...")
        for i in range(1):
            model = finetune()
            # 终止条件2；终止条件1还没实现!!!#编写自己的回调函数
            # 2017.5.22迭代一期，进行一次测试集的输出，并加上相应的损失
            ES = EarlyStopping(monitor='loss', min_delta=0.01, patience=0, verbose=1, mode='auto')
            history = model.fit([np.asarray(x_train), np.asarray(x_train_coord)], np.asarray(Y_label),
                                verbose=0, batch_size=10,
                                epochs=20,
                                callbacks=[ES]
                                )
            model.save_weights('my_model.h5')
        print("Done...")
        print("Training Time:", time.time() - t0)

    def estimate():
        t0 = time.time()

        print("Estimate...")
        model = finetune()
        model.load_weights("my_model.h5")

        Y_label = model.predict([np.asarray(x_test), np.asarray(x_test_coord)], batch_size=100, verbose=0)
        print("Done...")
        print("Estimation Time:", time.time() - t0)