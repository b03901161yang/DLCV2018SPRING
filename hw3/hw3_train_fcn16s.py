import sys
import numpy as np
import csv
import keras
from keras.models import Sequential
from keras.layers import Input, Dropout, Dense, Flatten, Add
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint
#2313 training
def readTrainDataLabel(dataPath):

    sat_image = []
    msk_image = []
    for n in range(2313):
        sat_im_tmp = Image.open(os.path.join(dataPath, str(n).zfill(4))+'_sat.jpg')
        sat_image.append(np.array(sat_im_tmp))
        sat_im_tmp.close()
        msk_im_tmp = Image.open(os.path.join(dataPath, str(n).zfill(4))+'_mask.png')
        msk_image.append(np.array(msk_im_tmp))
        msk_im_tmp.close()
    
    sat_image_arr = np.array(sat_image)
    msk_image_arr = np.array(msk_image)
    print(sat_image_arr.shape)
    return sat_image_arr, msk_image_arr

def parseMask(msk_image):
    msk_image_category = np.zeros((msk_image.shape[0], msk_image.shape[1], msk_image.shape[2], 7), dtype='uint8')
    for n in range(msk_image.shape[0]):
        mask0 = np.all(msk_image[n,:,:,:] == (0, 255, 255), axis=-1)
        #print('mask shape', mask.shape)
        msk_image_category[n, mask0, 0] = 1
        mask1 = np.all(msk_image[n,:,:,:] == (255, 255, 0), axis=-1)
        msk_image_category[n, mask1, 1] = 1
        mask2 = np.all(msk_image[n,:,:,:] == (255, 0, 255), axis=-1)
        msk_image_category[n, mask2, 2] = 1
        mask3 = np.all(msk_image[n,:,:,:] == (0, 255, 0), axis=-1)
        msk_image_category[n, mask3, 3] = 1
        mask4 = np.all(msk_image[n,:,:,:] == (0, 0, 255), axis=-1)
        msk_image_category[n, mask4, 4] = 1
        mask5 = np.all(msk_image[n,:,:,:] == (255, 255, 255), axis=-1)
        msk_image_category[n, mask5, 5] = 1
        mask6 = np.all(msk_image[n,:,:,:] == (0, 0, 0), axis=-1)
        msk_image_category[n, mask6, 6] = 1

    return msk_image_category

def writePredict(predictMsk, dataPath):
    for n in range(predictMsk):
        os.path.join(dataPath, str(n).zfill(4))+'_sat.jpg' 
        img_tmp = Image.fromarray(predictMsk[n,:,:])
        img_tmp.save(os.path.join(dataPath, str(n).zfill(4))+'_sat.jpg' )
        img_tmp.close()
    return predictMsk

    
########################    

dataPath_train = sys.argv[1] #'hw3-train-validation/train'
#dataPath_test  = 'predictMsk'
x_train, y_train_tmp = readTrainDataLabel(dataPath_train)

print('x train shape ',x_train.shape)
print('y train shape ',y_train_tmp.shape)

#print('x train [0] ',x_train[0,:,:,:])
#print('y train tmp[0] ',y_train_tmp[0,:,:,:])

y_train = parseMask(y_train_tmp)
print('y train shape after categorize',y_train.shape)
#print('y train [23, 10, 10, :] ',y_train[23,10,10,:])
########################################### set parameters here
batch_size = 4
num_classes = 7#0-6
epochs = 50
lr = 0.001
############################################
input_shape = (512, 512, 3)

x_train = x_train.astype('float32')
x_train /= 255
########################################################

def FCN_32(input_shape=None, num_classes=2):
    train_able = False
    img_input = Input(shape=input_shape)
    #block1
    x = Conv2D(64, (3, 3), activation='relu',padding='same', name='block1_conv1', trainable=train_able)(img_input)
    x = Conv2D(64, (3, 3), activation='relu',padding='same', name='block1_conv2', trainable=train_able)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #block2
    x = Conv2D(128, (3, 3), activation='relu',padding='same', name='block2_conv1', trainable=train_able)(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='same', name='block2_conv2', trainable=train_able)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #block3
    x = Conv2D(256, (3, 3), activation='relu',padding='same', name='block3_conv1', trainable=train_able)(x)
    x = Conv2D(256, (3, 3), activation='relu',padding='same', name='block3_conv2', trainable=train_able)(x)
    x = Conv2D(256, (3, 3), activation='relu',padding='same', name='block3_conv3', trainable=train_able)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x_pool3 = x
    #block4
    x = Conv2D(512, (3, 3), activation='relu',padding='same', name='block4_conv1', trainable=train_able)(x)
    x = Conv2D(512, (3, 3), activation='relu',padding='same', name='block4_conv2', trainable=train_able)(x)
    x = Conv2D(512, (3, 3), activation='relu',padding='same', name='block4_conv3', trainable=train_able)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x_pool4 = x
    #block5

    x = Conv2D(512, (3, 3), activation='relu',padding='same', name='block5_conv1', trainable=train_able)(x)
    x = Conv2D(512, (3, 3), activation='relu',padding='same', name='block5_conv2', trainable=train_able)(x)
    x = Conv2D(512, (3, 3), activation='relu',padding='same', name='block5_conv3', trainable=train_able)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x_pool5 = x

    vgg  = Model(  img_input , x  )
    vgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)

    o = x_pool5
    o = Conv2D(4096, (7, 7), activation='relu', padding='same',name='fc1')(o)
    o = Dropout(0.5)(o)
    o = Conv2D(4096, (1, 1), activation='relu', padding='same',name='fc2')(o)
    o = Dropout(0.5)(o)
    o = Conv2D(num_classes, (1, 1), kernel_initializer='he_normal', activation = 'linear' , padding='valid', name='fc3')(o)

    o_up2 = Conv2DTranspose(filters = num_classes, kernel_size = (4, 4), strides = (2, 2), padding='same', use_bias = False, name='up2')(o)
    
    x_pool4 = Conv2D(num_classes, (1, 1), kernel_initializer='he_normal', activation = 'linear' , padding='valid',name='conv_pool4')(x_pool4)
    # o_up2, x_pool4 = crop(o_up2, x_pool4, input img)
    o_fuse_pool4 = Add()([o_up2, x_pool4])
    o_up16 = Conv2DTranspose(filters = num_classes, kernel_size = (32, 32), strides = (16, 16), padding='same', activation='softmax', use_bias = False,name='up16')(o_fuse_pool4)
    model_fcn = Model(img_input, o_up16)

    return model_fcn

#############################################
checkpointer = ModelCheckpoint(filepath='fcn16s.h5', verbose=1)

model = FCN_32(input_shape = input_shape, num_classes = num_classes)
model.summary()

finetune = 0
if(finetune):
    model.load_weights('fcn16s.h5')
    lr = 0.0001

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr = lr),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          verbose=1, 
          callbacks=[checkpointer])

print("model saved")

