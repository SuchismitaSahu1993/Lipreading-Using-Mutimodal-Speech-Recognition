import librosa
import numpy as np
from queue import PriorityQueue
import os

import warnings
import numpy as np
from keras.layers import LSTM
from keras.layers import Dense, Activation, dot
from keras.layers import Input
from keras.models import Model
from collections import defaultdict
from keras.layers import concatenate
from keras.layers import Bidirectional
from keras.layers import TimeDistributed

from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint, EarlyStopping

phoneme_set_list = [
            'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow', #13 phns
            'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx',  #14 phns
            'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil' #12 pns
        ]

phoneme_int = {}
int_phoneme = {}
label = 1
for phoneme in phoneme_set_list:
    phoneme_int[phoneme] = label
    int_phoneme[label] = phoneme
    label += 1

def one_hot_transform(y):
    one_hot = []
    for label in y:
        temp = [0 for _ in range(39)]
        temp[phoneme_int[label] - 1] = 1
        one_hot.append(temp)
    return one_hot

def inverse_one_hot_transform(y, k=1):
    inverse_labels = []
    # for vector in y:
    pq = PriorityQueue()
    for i, val in enumerate(y):
        pq.put((-val, i+1))
    temp = []
    while k:
        idx = pq.get()[1]
        phn = int_phoneme[idx]
        temp.append(phn)
        k -= 1
    inverse_labels.append(temp)
    return inverse_labels

def collect_audio_files(folder):

    wav_files  = []

    for path, _, files in os.walk(folder):
        for f in files:
            if f.endswith('wav'):
                wav_files.append(path + '/' + f)

    return wav_files

def create_audio_dataset_when_labelled(filelist):
    X, y= [], []
    for i, filename in enumerate(filelist):
        try:
            print(f'{i+1} / {len(filelist)}')
            phn_file = filename[:-3]+'phn'

            audio, orig_sr = librosa.load(filename)
            audio, sr = resample_audio(phn_file=phn_file, audio=audio, orig_sr=orig_sr)

            phon_split_audio, phn_list = create_mfcc_labels(phn_file=phn_file, audio=audio, sample_rate=sr)
            phon_split_audio = np.array(phon_split_audio)
            X.extend(phon_split_audio)
            y.extend(phn_list)
        except:
            pass

    X = np.array(X)

    return X, y

def resample_audio(phn_file, audio, orig_sr):

    with open(phn_file) as f:
        for line in f:
            _, stop, _ = line.strip('\n').split()

    target_sr = 16000
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio, target_sr

def create_mfcc_labels(phn_file, audio, sample_rate):

    phon_split_audio, phn_list = [], []

    with open(phn_file) as f:
        for line in f:
            start, stop, phn = line.strip('\n').split()
            start, stop = int(start), int(stop)
            temp_x = audio[start:stop]
            mfcc = librosa.feature.mfcc(y=temp_x, sr=sample_rate, n_mfcc=13).T
            phon_split_audio.extend(mfcc)
            temp_y = [phn]*mfcc.shape[0]
            phn_list.extend(temp_y)

    return phon_split_audio, phn_list

def create_audio_test_dataset(filename):
    phon_split_audio = []
    audio, orig_sr = librosa.load(filename)
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
    mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=13).T 
    phon_split_audio.extend(mfcc)
    phon_split_audio = np.array(phon_split_audio)

    return phon_split_audio

def preprocessing_for_rnn(phon_split_audio, stack_size=5, phn_list=None):
        if phn_list:
            return stack_for_train(stack_size=stack_size, phon_split_audio=phon_split_audio,phn_list=phn_list)
        else:
            return stack_for_test(stack_size=stack_size, phon_split_audio=phon_split_audio)

def stack_for_train(stack_size, phon_split_audio, phn_list):
    X, y = [], []
    k = 0
    while k + stack_size < phon_split_audio.shape[0]:
        temp_x, temp_y = phon_split_audio[k:k+stack_size], [phn_list[k+stack_size//2 + 1]] * stack_size        
        X.append(np.array(temp_x))
        temp_y = np.array(one_hot_transform(temp_y))
        y.append(temp_y)
        k += 1
    X = np.array(X)
    y = np.array(y)
    return X, y

def stack_for_test(stack_size, phon_split_audio):
    X = [] 
    k = 0
    while k + 5 < phon_split_audio.shape[0]:
        temp_x = phon_split_audio[k:k+5]
        X.append(np.array(temp_x))
        k += 1
    return X

def pad_zeros(X_audio=None, y_audio=None, X_video=None, y_video=None):
    if not np.any(X_audio):
        n = X_video.shape[1]
        X_audio = np.zeros((n, 1, 13))
        y_audio = np.zeros((n, 1, 39))
    elif not np.any(X_video):
        m = X_audio.shape[1]
        X_video = np.zeros((m, 120, 120, 1))
        y_video = np.zeros((m, 39))
    else:
        m = X_audio.shape[0] - X_video.shape[0]
        pad_zeros_x = np.zeros((m, 120, 120, 1))
        pad_zeros_y = np.zeros((m, 39))
        X_video = np.concatenate((X_video, pad_zeros_x), axis=0)
        y_video = np.concatenate((y_video, pad_zeros_y), axis=0)

    return X_audio, y_audio, X_video, y_video

def define_audio_network(stack_size):
    InputLayer = Input(shape= (stack_size,13))
    encoder, h_s, c_s = LSTM(256, return_sequences=True, return_state=True)(InputLayer)
    decoder = LSTM(256, return_sequences=True)(encoder, initial_state = [c_s, c_s])
    attention = dot([decoder, encoder], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)
    context = dot([attention, encoder], axes=[2,1])
    decoder_combined_context = concatenate([context, decoder])
    output = Dense(256, activation="tanh")(decoder_combined_context) 
    output = Dense(39, activation="softmax")(output)
    model = Model(inputs=InputLayer, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == '__main__':

    dataset_train = collect_audio_files('../../data/TCDTIMIT/audio/lipspeakers')
    test_file_list, train_file_list = [], []

    test_list = [f'{dataset_train}/Lipspkr1_sa1.wav', f'{dataset_train}/Lipspkr1_sa2.wav',
                     f'{dataset_train}/Lipspkr2_sa2.wav', f'{dataset_train}/Lipspkr2_sa2.wav', 
                     f'{dataset_train}/Lipspkr3_sa2.wav', f'{dataset_train}/Lipspkr3_sa2.wav']
    for filename in dataset_train:
        if 'sa1' not in filename and 'sa2' not in filename:
            train_file_list.append(filename)
        else:
            test_file_list.append(filename)

    print(f'{len(train_file_list)} files for train and {len(test_file_list)} for test')

    # dataset_test = collect_audio_files('../../data/TCDTIMIT/audio/test')

    # for filename in dataset_test:
    #         test_file_list.append(filename)

    stack_size = 9
    val_file, train_file_list = train_file_list[0], train_file_list[1:]
    X_val, y_val = create_audio_dataset_when_labelled([val_file])
    X_val, y_val = preprocessing_for_rnn(X_val, stack_size=stack_size, phn_list=y_val)
    X_train, y_train = create_audio_dataset_when_labelled(train_file_list)
    X_train, y_train = preprocessing_for_rnn(X_train, stack_size=stack_size, phn_list=y_train)
    model = define_audio_network(stack_size=stack_size)

    # plot_model(model, to_file='audio_model.png', show_shapes=True)

    filepath = './audio_lipspeaker_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping =  EarlyStopping(patience=3, verbose=1)
    callbacks_list = [checkpoint, early_stopping]
    model.fit(X_train, y_train, epochs=50,validation_data=(X_val, y_val), callbacks=callbacks_list)
    for file in test_file_list:
        print(f'Filename : {file}')
        X_test, y_test = create_audio_dataset_when_labelled([file])
        X_test, y_test = preprocessing_for_rnn(X_test, stack_size=stack_size, phn_list=y_test)
        y_pred = model.predict(X_test)

        print(model.evaluate(X_test, y_test))

        one_count = 0
        three_count = 0
        five_count = 0
        test = []
        prediction = []
        true_seq = []
        pred_seq = []
        for true, pred in zip(y_test, y_pred):
            t = list(true[stack_size//2 + 1])
            p = list(pred[stack_size//2 + 1])
            actual = inverse_one_hot_transform(t)
            pred_one = inverse_one_hot_transform(p)
            pred_three = inverse_one_hot_transform(p, k=3)
            pred_five = inverse_one_hot_transform(p, k=5)

            if actual == pred_one:
                one_count += 1
            if actual[0][0] in pred_three[0]:
                three_count += 1
            if actual[0][0] in pred_five[0]:
                five_count += 1

            test.append(np.array(t))
            prediction.append(np.array(p))
            true_seq.append(actual[0][0])
            pred_seq.append(pred_one[0][0])

        print('Top - 1', one_count / true.shape[0])
        print('Top - 3', three_count / true.shape[0])
        print('Top - 5', five_count / true.shape[0])
        for i, j in zip(true_seq, pred_seq):
            print(i, j)

        file_path = f'{os.getcwd()}/audio_lipspeaker_model.txt' 
        with open(file_path, 'a+') as f:
            result = 'Top-1 ' + str(one_count / true.shape[0]) + ' Top-3 ' + str(three_count / true.shape[0]) + ' Top-5 ' + str(five_count / true.shape[0])
            f.write(result)
            f.write("\n")





