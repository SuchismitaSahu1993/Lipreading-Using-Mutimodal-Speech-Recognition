import librosa
import numpy as np
from queue import PriorityQueue
import os
import json
import pickle as pkl
import pdb
import warnings
import h5py
from PIL import Image
import numpy as np
import keras.backend as K
from collections import defaultdict
from sklearn.externals import joblib
from keras.models import Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.layers import *
from phonemeSet import *
from buildModel import *

phoneme_int = {}
int_phoneme = {}
label = 1
for phoneme in phoneme_set_39_list:
    phoneme_int[phoneme] = label
    int_phoneme[label] = phoneme
    label += 1

def one_hot_transform(y):
    one_hot = []
    for label in y:
        temp = [0] * 39
        temp[phoneme_int[label]-1] = 1
        one_hot.append(temp)
    return one_hot

def inverse_one_hot_transform(y, k=1):
    inverse_labels = []
    for vector in y:
        temp_k = k
        pq = PriorityQueue()
        for i, val in enumerate(vector):
            pq.put((-val, i+1))
        temp = []
        while k:
            idx = pq.get()[1]
            phn = int_phoneme[idx]
            temp.append(phn)
            k-=1
        inverse_labels.append(temp)
        k = temp_k
    return inverse_labels

def collect_video_files(data_type, resp, root_dir):

    if resp == 'lipspeakers':
        speakerDir = root_dir+'/lipspeakers'
    elif resp == 'volunteers':
        speakerDir = root_dir+'/volunteers'
    else:
        speakerDir = root_dir

    fileName = f'{data_type}_dict.json'
    data = open(speakerDir+os.sep+fileName).read()
    data_dict = json.loads(data)

    with open(f'{speakerDir}/data_phn.txt', 'r') as f:
        lines = f.readlines()
    video_list = [l.strip() for l in lines]

    return data_dict, video_list, speakerDir

def create_audio_dataset(filelist, train=True):
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

    target_sr = (int(stop) / audio.shape[0]) * orig_sr
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

def preprocessing_for_rnn(phon_split_audio, phn_list=None, stack_size=5):
        if phn_list:
            return stack_for_train(stack_size=stack_size, phon_split_audio=phon_split_audio,phn_list=phn_list)
        else:
            return stack_for_test(stack_size=stack_size, phon_split_audio=phon_split_audio)

def stack_for_train(stack_size, phon_split_audio, phn_list):
    X, y = [], []
    k = 0
    while k + stack_size < phon_split_audio.shape[0]:
        temp_x, temp_y = phon_split_audio[k:k+stack_size], [phn_list[k+stack_size//2]] * stack_size        
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
    while k + stack_size < phon_split_audio.shape[0]:
        temp_x = phon_split_audio[k:k+stack_size]
        X.append(np.array(temp_x))
        k += 1
    X = np.array(X)
    return X

def get_test_data(video, data_labels, data_dir):
    spk,vdo  = video.split('_')[:2]
    image_name = spk+'_'+vdo+'_'
    list_IDs_temp = sorted([image_file for image_file in list(data_labels) if image_name in image_file])
    time_steps = len(list_IDs_temp)
    X = np.zeros(( time_steps,120,120,1))
    Y=[]
    for i, ID in enumerate(list_IDs_temp):
        im = np.array(Image.open(data_dir+'/'+ID), dtype=np.uint8)
        im = np.expand_dims(im,3)
        im = im.astype('float32')/255
        X[i] = im
        #y[i] = phoneme_int[data_labels[ID]]
        Y.append(data_labels[ID])
    y = np.array(one_hot_transform(np.array(Y)))
    x_test = X.reshape(1, X.shape[0], 120, 120, 1)
    y_test = y.reshape(1, y.shape[0], 39)
    return x_test, y_test    

def evaluate_model(epoch,  model, x_test, y_test):
    file = f"Epoch {epoch} metrics.txt"
    
    file_path = f'{os.getcwd()}/{file}' 

    y_pred = model.predict(x_test)
    for y_p, y in zip(y_pred, y_test):        
        top1 = inverse_one_hot_transform(y_p, k=1)
        top3 = inverse_one_hot_transform(y_p, k=3)
        top5 = inverse_one_hot_transform(y_p, k=5)
        top1_acc, top3_acc, top5_acc = 0.0,0.0,0.0 
        for y_t, tp1 in zip(y, top1):
            print(int_phoneme[y_t[0]], " \t", tp1)
            if int_phoneme[y_t[0]] in tp1:
                top1_acc+=1
        for y_t, tp3 in zip(y, top3):
            if int_phoneme[y_t[0]] in tp3:
                top3_acc+=1
        for y_t, tp5 in zip(y, top5):
            if int_phoneme[y_t[0]] in tp5:
                top5_acc+=1

        y_len = len(top1)
        
        top1_acc/=y_len
        top3_acc/=y_len
        top5_acc/=y_len

        top1_acc = round(top1_acc*100,2)
        top3_acc = round(top3_acc*100,2)
        top5_acc = round(top5_acc*100,2)

        with open(file_path, 'a+') as f:
            f.write(f"Top-1 accuracy: {top1_acc}% \t Top-3 accuracy: {top3_acc}% \t Top-5 accuracy: {top5_acc}%")
            f.write("\n")
        print(f"Top-1 accuracy: {top1_acc}%")
        print(f"Top-3 accuracy: {top3_acc}%")
        print(f"Top-5 accuracy: {top5_acc}%")
        return 1-top1_acc, top1_acc

def fcNetwork():
    InputLayer = Input(shape = (78,))
    dense = Dense(1000, activation="relu")(InputLayer) 
    output = Dense(39, activation="softmax")(dense)
    model = Model(inputs=InputLayer, outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def fix_length(audio, video):
    period  = audio.shape[0]//video.shape[0]
    new_video = video.repeat(period,axis = 0)
    if new_video.shape != audio.shape:
        rep = audio.shape[0] - new_video.shape[0]
        pred = np.array([new_video[-1],]*rep)
        new_video = np.concatenate((new_video, pred), axis=0)
    return new_video

if __name__ == '__main__':
    
    print(f"Collecting data..")

    #video_path = os.path.join(os.path.expanduser('../../TCDTIMIT/video'))
    video_path = os.path.join(os.path.expanduser('../../data/TCDTIMIT/video'))
    spkr_type = 'lipspeakers'

    lip_data_dict, lip_video_list, lip_data_dir = collect_video_files('data', 'lipspeakers',video_path)

    trainspeakerList = ['Lipspkr1','Lipspkr2','Lipspkr3' ]
    testspeakerList = ['Lipspkr3']
    # trainVolunteerList = ['01M']
    testVolunteerList = ['01M']

    test_list = []
    val_list = []
    train_list  = []

    for speaker in trainspeakerList:
        train_list+=[k for k in lip_video_list if speaker in k][:-2]
    for speaker in testspeakerList:         
        test_val_list=[k for k in lip_video_list if speaker in k][-2:]
        half = len(test_val_list)//2
        test_list += test_val_list[:half]
        #val_list += test_val_list[half:]
    
    #audio_path = os.path.join(os.path.expanduser(f'../../TCDTIMIT/audio/{spkr_type}')) 
    audio_path = os.path.join(os.path.expanduser(f'../../data/TCDTIMIT/audio/{spkr_type}'))        
    audioTrainList =[]
    for file in train_list:
        file = file.split('.')[0].split('_')[:2]
        filename = file[0]+'_'+file[1]
        audioTrainList.append(audio_path+'/'+filename+'.wav')


    # spkr_type = 'volunteers'
    # #print(f"Collecting {spkr_type} data..")
    # vol_data_dict, vol_video_list, vol_data_dir = collect_video_files('data', 'volunteers', video_path)
        

    # # if spkr_type == 'volunteers':
    # #     for vol in trainVolunteerList:      
    # #         train_list+=[k for k in vol_video_list if vol in k][2:]      
    # for vol in testVolunteerList:
    #     test_val_list=[k for k in vol_video_list if vol in k][-2:]
    #     half = len(test_val_list)//2
    #     test_list += test_val_list[:half]
        # val_list += test_val_list[half:]

    #audio_path = os.path.join(os.path.expanduser(f'../../TCDTIMIT/audio/{spkr_type}'))        
    audio_path = os.path.join(os.path.expanduser(f'../../data/TCDTIMIT/audio/{spkr_type}'))  
    audioTestList =[]
    for file in test_list:
        file = file.split('.')[0].split('_')[:2]
        filename = file[0]+'_'+file[1]
        audioTestList.append(audio_path+'/'+filename+'.wav')


    audio_model_path = f'{os.getcwd()}/audio_model_final.hdf5' 
    video_model_path = f'{os.getcwd()}/video_model_best.hdf5' 

    audio_model = load_model(audio_model_path)
    video_model = load_model(video_model_path)

    model = fcNetwork()
    X = []
    y = []
    TRAIN = 1
    if TRAIN == 1:
        for audio_file, video_file in zip(audioTrainList, train_list):
            try:
                print(f"Collecting data from {audio_file}...")
                X_test, y_test = create_audio_dataset([audio_file])
                X_audio_test, y_audio_test = preprocessing_for_rnn(X_test, phn_list=y_test,  stack_size=9)
                audio_pred = audio_model.predict(X_audio_test,verbose=0)
                new_audio_pred = []
                new_audio_test = []
                for pred, test in zip(audio_pred, y_audio_test):
                    new_audio_pred.append(list(pred[5]))
                    new_audio_test.append(list(test[5]))
                new_audio_pred = np.vstack(new_audio_pred)
                new_audio_test = np.vstack(new_audio_test)

                X_video_test, y_video_test = get_test_data(video_file, lip_data_dict, lip_data_dir)
                video_pred = video_model.predict(X_video_test,verbose=0)

                video_pred = fix_length(new_audio_pred, video_pred[0])
                video_test = fix_length(new_audio_test, y_video_test[0])
                
                X_train = np.concatenate((new_audio_pred, video_pred), axis=1)
                y_train = video_test
                X.append(X_train)
                y.append(y_train)
            except:
                print(f"error in {audio_file} - {video_file}")

        X = np.vstack(X)
        y = np.vstack(y)

        hf = h5py.File(f"{os.getcwd()}/X_train.h5", 'w')
        hf.create_dataset('X_train', data=X)

        hf = h5py.File(f"{os.getcwd()}/y_train.h5", 'w')
        hf.create_dataset('y_train', data=y)

        hf.close()
    else:
        print("Creating Train data...")
        hf = h5py.File(f"{os.getcwd()}/X_train.h5", 'r')
        X= np.array(hf.get('X_train'))
        
        hf.close()
        hf = h5py.File(f"{os.getcwd()}/y_train.h5", 'r')
        y= np.array(hf.get('y_train'))
        
        hf.close()

    print("Training Model...")
    filepath=f"{os.getcwd()}/combined_best.hdf5"
    earlyStop = EarlyStopping(patience=5)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,earlyStop]
    model.fit(X,y, epochs=20, callbacks= callbacks_list)

    for audio_file, video_file in zip(audioTestList, test_list):
        X_test, y_test = create_audio_dataset([audio_file])
        X_audio_test, y_audio_test = preprocessing_for_rnn(X_test, phn_list=y_test,  stack_size=9)
        audio_pred = audio_model.predict(X_audio_test)
        new_audio_pred = []
        new_audio_test = []
        for pred, test in zip(audio_pred, y_audio_test):
            new_audio_pred.append(list(pred[5]))
            new_audio_test.append(list(test[5]))
        new_audio_pred = np.vstack(new_audio_pred)
        new_audio_test = np.vstack(new_audio_test)

        X_video_test, y_video_test = get_test_data(video_file, lip_data_dict, lip_data_dir)
        video_pred = video_model.predict(X_video_test)

        video_pred = fix_length(new_audio_pred, video_pred[0])
        video_test = fix_length(new_audio_test, y_video_test[0])

        X_train = np.concatenate((new_audio_pred, video_pred), axis=1)
        print("Evaluating Model...")
        res = model.evaluate(X_train, video_test)
        file_path = f'{os.getcwd()}/combined_metrics.txt' 
        print(res)
        with open(file_path, 'a+') as f:
            f.write(f"Accuracy: {res[1]}% \t Loss: {res[0]}%")
            f.write("\n")








