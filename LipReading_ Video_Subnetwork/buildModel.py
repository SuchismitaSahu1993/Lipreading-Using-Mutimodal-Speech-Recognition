import numpy as np
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception

def resnet_lstm():
    # create the base pre-trained model
    resn_model = ResNet50( include_top=False, weights='imagenet', input_shape= (120,120, 3),pooling='avg' )

    input_layer = Input(shape=(30, 120,120, 3))

    
    curr_layer = TimeDistributed(resn_model)(input_layer)
    curr_layer = TimeDistributed(Flatten())(curr_layer)
    #x = Reshape((30,2048))(resn_model.output)
    #curr_layer = Reshape(target_shape=(30, 2048))(curr_layer)
    lstm_out = LSTM(128)(curr_layer)
    
    predictions = Dense(39, activation='softmax')(lstm_out)

    model = Model(inputs=resn_model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary()

    return model

def cifar10():
    inputs = Input(shape=(None, 120,120,1))
    conv1 = TimeDistributed(Conv2D(128, 3))(inputs)
    zp1 = TimeDistributed(ZeroPadding2D(1))(conv1)
    bn1 = TimeDistributed(BatchNormalization())(zp1)
    act1 = TimeDistributed(Activation('relu'))(bn1)

    conv2 = TimeDistributed(Conv2D(128, 3))(act1)
    zp2 = TimeDistributed(ZeroPadding2D(1))(conv2)
    mp1 = TimeDistributed(MaxPool2D(2))(zp2)
    bn2 = TimeDistributed(BatchNormalization())(mp1)
    act2 = TimeDistributed(Activation('relu'))(bn2)

    conv3 = TimeDistributed(Conv2D(256, 3))(act2)
    zp3 = TimeDistributed(ZeroPadding2D(1))(conv3)
    bn3 = TimeDistributed(BatchNormalization())(zp3)
    act3 = TimeDistributed(Activation('relu'))(bn3)

    conv4 = TimeDistributed(Conv2D(256, 3))(act3)
    zp4 = TimeDistributed(ZeroPadding2D(1))(conv4)
    mp2 = TimeDistributed(MaxPool2D(2))(zp4)
    bn4 = TimeDistributed(BatchNormalization())(mp2)
    act4 = TimeDistributed(Activation('relu'))(bn4)

    conv5 = TimeDistributed(Conv2D(512, 3))(act4)
    zp5 = TimeDistributed(ZeroPadding2D(1))(conv5)
    bn5 = TimeDistributed(BatchNormalization())(zp5)
    act5 = TimeDistributed(Activation('relu'))(bn5)

    conv6 = TimeDistributed(Conv2D(256, 3))(act5)
    zp6 = TimeDistributed(ZeroPadding2D(1))(conv6)
    mp3 = TimeDistributed(MaxPool2D(2))(zp6)
    bn6 = TimeDistributed(BatchNormalization())(mp3)
    act6 = TimeDistributed(Activation('relu'))(bn6)

    # d1 = TimeDistributed(Dense(256))(act6)
    # bn7 = TimeDistributed(BatchNormalization())(d1)
    # act7 = TimeDistributed(Activation('relu'))(bn7)

    FlattenedOutput = TimeDistributed(Flatten())(act6)
    
    LSTM1 = LSTM(128, return_sequences=True)(FlattenedOutput)

    finalOutput = TimeDistributed(Dense(39, activation='softmax'))(LSTM1)

    model = Model(inputs, finalOutput)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary() 
    return model

def cifar10_v2():

    inputs = Input(shape=(None, 120,120,1))
    conv1 = TimeDistributed(Conv2D(192, 5))(inputs)
    zp1 = TimeDistributed(ZeroPadding2D(2))(conv1)
    conv2 = TimeDistributed(Conv2D(160, 1,padding='same'))(zp1)
    conv3 = TimeDistributed(Conv2D(96, 1, padding='same'))(conv2)
    mp1 =TimeDistributed(MaxPooling2D(pool_size=3, strides=2))(conv3)
    dp1 = TimeDistributed(Dropout(0.5))(mp1)

    conv4 = TimeDistributed(Conv2D(192, 5))(dp1)
    zp2 = TimeDistributed(ZeroPadding2D(2))(conv4)
    conv5 = TimeDistributed(Conv2D(192, 1, padding='same'))(zp2)
    conv6 = TimeDistributed(Conv2D(192, 1, padding='same'))(conv5)
    ap1 =TimeDistributed(AveragePooling2D(pool_size=3, strides=2))(conv6)
    dp2 = TimeDistributed(Dropout(0.5))(ap1)
    
    conv7 = TimeDistributed(Conv2D(192, 3))(dp2)
    zp3 = TimeDistributed(ZeroPadding2D(1))(conv7)
    conv8 = TimeDistributed(Conv2D(192, 1, padding='same'))(zp3)
    conv9 = TimeDistributed(Conv2D(10, 1, padding='same'))(conv8)
    ap2 =TimeDistributed(AveragePooling2D(pool_size=8))(conv9)
        
    FlattenedOutput = TimeDistributed(Flatten())(ap2)
    
    LSTM1 = LSTM(256, return_sequences=True)(FlattenedOutput)

    outputs = TimeDistributed(Dense(39, activation='softmax'))(LSTM1)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary()
    
    return model

def VisemeWASnet(input_shape, n_classes):
    
    InputLayer = Input(shape=input_shape)

    conv1 = TimeDistributed(Conv2D(64, kernel_size = (3,3), name='conv1' ,padding='same', activation='relu'))(InputLayer)
    pool1 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv1)
    Bn2 = TimeDistributed(BatchNormalization())(pool1)

    conv2 = TimeDistributed(Conv2D(128, kernel_size =(3,3), name='conv2', activation='relu', padding ='same'))(Bn2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv2)
    Bn3 = TimeDistributed(BatchNormalization())(pool2)

    conv3 = TimeDistributed(Conv2D(256, kernel_size = (3,3), name='conv3',padding='same', activation='relu'))(Bn3)
    MaxPool = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv3) 
    Bn4 = TimeDistributed(BatchNormalization())(MaxPool)

    flatten = TimeDistributed(Flatten())(Bn4)

    lstm1 = LSTM(128, return_sequences = True)(flatten) 
    lstm = LSTM(128, return_sequences = True, name = 'lstm')(lstm1) 
    
    output = TimeDistributed(Dense(n_classes, activation='softmax'))(lstm)

    model = Model(InputLayer, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary()
    
    return model

def WASnet(input_shape, n_classes):
    
    InputLayer = Input(shape=input_shape)

    conv1 = TimeDistributed(Conv2D(96, kernel_size = (3,3), name='conv1' ,padding='same', activation='relu'))(InputLayer)
    pool1 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv1)
    Bn2 = TimeDistributed(BatchNormalization())(pool1)

    conv2 = TimeDistributed(Conv2D(256, kernel_size =(3,3), name='conv2', activation='relu', padding ='same'))(Bn2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv2)
    Bn3 = TimeDistributed(BatchNormalization())(pool2)

    conv3 = TimeDistributed(Conv2D(512, kernel_size = (3,3), name='conv3',padding='same', activation='relu'))(Bn3)
    conv4 = TimeDistributed(Conv2D(512, kernel_size = (3,3), name='conv4',padding='same', activation='relu'))(conv3)
    conv5 = TimeDistributed(Conv2D(512, kernel_size = (3,3), name='conv5',padding='same', activation='relu'))(conv4)
    MaxPool = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv5) 
    Bn4 = TimeDistributed(BatchNormalization())(MaxPool)

    pool5 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(Bn4) 

    flatten = TimeDistributed(Flatten())(pool5)

    lstm1 = LSTM(128, return_sequences = True)(flatten) 
    lstm = LSTM(128, return_sequences = True, name = 'lstm')(lstm1) 
    
    output = TimeDistributed(Dense(n_classes, activation='softmax'))(lstm)

    model = Model(InputLayer, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary()
    
    return model

def googleNet():

    epsilon = 1e-4

    InputLayer = Input(shape=(None, 120,120,1))

    conv1 = TimeDistributed(Conv2D(128, kernel_size = (3,3)))(InputLayer)
    zp1 = TimeDistributed(ZeroPadding2D(1))(conv1)
    mp1 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(zp1) 
    bn1 = TimeDistributed(BatchNormalization(epsilon = epsilon))(mp1)
    act1 = TimeDistributed(Activation('relu'))(bn1)

    conv2 = TimeDistributed(Conv2D(256, kernel_size = (3,3), strides = (2,2)))(act1)
    zp2 = TimeDistributed(ZeroPadding2D(1))(conv2)
    mp2 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(zp2) 
    bn2 = TimeDistributed(BatchNormalization(epsilon = epsilon ))(mp2)   
    act2 = TimeDistributed(Activation('relu'))(bn2)

    conv3 = TimeDistributed(Conv2D(512, kernel_size =(3,3)))(act2)
    zp3 = TimeDistributed(ZeroPadding2D(1))(conv3)
    act3 = TimeDistributed(Activation('relu'))(zp3)

    conv4 = TimeDistributed(Conv2D(512, kernel_size = (3,3)))(act3)
    zp4 = TimeDistributed(ZeroPadding2D(1))(conv4)
    act4 = TimeDistributed(Activation('relu'))(zp4)

    conv5 = TimeDistributed(Conv2D(512, kernel_size =(3,3)))(act4)
    zp5 = TimeDistributed(ZeroPadding2D(1))(conv5)
    mp3 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(zp5) 
    
    act5 = TimeDistributed(Activation('relu'))(mp3)

    FlattenedOutput = TimeDistributed(Flatten())(act5)
    
    LSTM1 = LSTM(128, return_sequences=True)(FlattenedOutput)
    
    FinalOutput = TimeDistributed(Dense(39, activation='softmax'))(LSTM1)

    model = Model(InputLayer, FinalOutput)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary()
    
    return model

def combined_network():
    # Inputs
    AudioInputLayer = Input(shape=(None,13))
    VideoInputLayer = Input(shape=(None, 120, 120, 1))

    # Audio LSTM
    encoder_a, hs_a, cs_a = LSTM(128, return_sequences=True, return_state=True)(AudioInputLayer)

    # Video Network
    conv1 = TimeDistributed(Conv2D(96, kernel_size = (3,3), name='conv1' ,padding='same', activation='relu'))(VideoInputLayer)
    pool1 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv1)
    Bn2 = TimeDistributed(BatchNormalization())(pool1)

    conv2 = TimeDistributed(Conv2D(256, kernel_size =(3,3), name='conv2', activation='relu', padding ='same'))(Bn2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv2)
    Bn3 = TimeDistributed(BatchNormalization())(pool2)

    conv3 = TimeDistributed(Conv2D(512, kernel_size = (3,3), name='conv3',padding='same', activation='relu'))(Bn3)
    conv4 = TimeDistributed(Conv2D(512, kernel_size = (3,3), name='conv4',padding='same', activation='relu'))(conv3)
    conv5 = TimeDistributed(Conv2D(512, kernel_size = (3,3), name='conv5',padding='same', activation='relu'))(conv4)
    MaxPool = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(conv5) 
    Bn4 = TimeDistributed(BatchNormalization())(MaxPool)

    pool5 = TimeDistributed(MaxPooling2D(pool_size = (2,2)))(Bn4) 

    flatten = TimeDistributed(Flatten())(pool5)

    encoder_v, hs_v, cs_v = LSTM(128, return_sequences = True,  return_state = True)(flatten)

    # Merge Audio and video layers
    encoder = concatenate([encoder_a, encoder_v])
    hs = concatenate([hs_a, hs_v])
    cs = concatenate([cs_a, cs_v])

    # To make the inputs compatible with attention inputs
    encoder = Dense(128)(encoder)
    hs = Dense(128)(hs)
    cs = Dense(128)(cs)

    # Decoder LSTM TODO modify with video parameters
    decoder = LSTM(128, return_sequences=True)(encoder, initial_state = [hs, cs])

    # Attention for audio
    attention_a = dot([decoder, encoder_a], axes=[2, 2])
    attention_a = Activation('softmax', name='attention_audio')(attention_a)
    context_a = dot([attention_a, encoder_a], axes=[2,1])
    decoder_combined_context_a = concatenate([context_a, decoder])

    # Attention for video
    attention_v = dot([decoder, encoder_v], axes=[2, 2])
    attention_v = Activation('softmax', name='attention_video')(attention_v)
    context_v = dot([attention_v, encoder_v], axes=[2,1])
    decoder_combined_context_v = concatenate([context_v, decoder])

    # # MLP for combining audio and video sides
    decoder_combined_context = concatenate([decoder_combined_context_a, decoder_combined_context_v])
    output = TimeDistributed(Dense(1000, activation="relu"))(decoder_combined_context) 

    finalOutput = TimeDistributed(Dense(39, activation="softmax"))(output)
    model = Model(inputs=[AudioInputLayer, VideoInputLayer], outputs=finalOutput)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def resnet():
    # create the base pre-trained model
    resn_model = ResNet50( include_top=False, weights='imagenet', pooling='avg', input_shape= (120,120,3) )
    for layer in resn_model.layers:
        layer.trainable=False

    x = Dense(1024, activation='relu')(resn_model.output)
    predictions = Dense(39, activation='softmax')(x)

    model = Model(inputs=resn_model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.summary()

    return model

if __name__ == "__main__":
    model = resnet()

