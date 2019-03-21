import os
import json
import pdb
import warnings
from PIL import Image
import numpy as np
import keras.backend as K
from queue import PriorityQueue
from keras.models import Model, Sequential, load_model
from phonemeSet import *
import buildModel as buildModel

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

def pad_zeros(X_audio=None, y_audio=None, X_video=None, y_video=None):
    if not np.any(X_audio):
        n = X_video.shape[1]
        X_audio = np.zeros((1, n, 13))
        y_audio = np.zeros((1, n, 39))
    elif not np.any(X_video):
        m = X_audio.shape[1]
        X_video = np.zeros((1, m, 120, 120, 1))
        y_video = np.zeros((1, m, 39))
    else:
        m = X_audio.shape[1] - X_video.shape[1]
        pad_zeros_x = np.zeros((1, m, 120, 120, 1))
        pad_zeros_y = np.zeros((1, m, 39))
        X_video = np.concatenate((X_video, pad_zeros_x), axis=1)
        y_video = np.concatenate((y_video, pad_zeros_y), axis=1)

    return X_audio, y_audio, X_video, y_video

def get_test_data(video, data_labels, data_dir):
    spk,vdo  = video.split('_')[:2]
    image_name = spk+'_'+vdo+'_'
    list_IDs_temp = sorted([image_file for image_file in list(data_labels) if image_name in image_file])
    time_steps = len(list_IDs_temp)
    X = np.zeros((time_steps,120,120,1))
    y = np.zeros((time_steps,1), dtype=int)
    for i, ID in enumerate(list_IDs_temp):
        im = np.array(Image.open(data_dir+'/'+ID), dtype=np.uint8)
        im = np.expand_dims(im,3)
        im = im.astype('float32')/255
        X[i] = im
        y[i] = phoneme_int[data_labels[ID]]
    x_test = X.reshape(1, X.shape[0], 120, 120, 1)
    y_test = y.reshape(1, y.shape[0], 1)
    return x_test, y_test, video

def get_train_data(video, data_labels,data_dir):
    spk,vdo = video.split('_')[:2]
    image_name = spk+'_'+vdo+'_'
    
    list_IDs_temp = sorted([image_file for image_file in list(data_labels) if image_name in image_file])
    X = np.zeros((len(list_IDs_temp), 120,120,1))
    #Y = np.zeros((len(list_IDs_temp),1))
    Y = []
    for i, ID in enumerate(list_IDs_temp):
        im = np.array(Image.open(data_dir+'/'+ID), dtype=np.uint8)
        im = np.expand_dims(im, 3)
        im = im.astype('float32')/255
        X[i] = im
        Y.append(data_labels[ID])
    y = np.array(one_hot_transform(np.array(Y)))
    x_train = X.reshape(1, X.shape[0], 120, 120, 1)
    y_train = y.reshape(1,y.shape[0], 39)
    return x_train, y_train

def train_model(val_list, train_list, data_dict, data_dir):
    input_shape = (None, 120,120,1)
    model = buildModel.WASnet(input_shape, 39)

    X_val, y_val, val_file = get_test_data(val_list[0], data_dict, data_dir)

    print(f"Training Model...")
    num_epochs = 20
    best_acc = 0.0
    for epoch in range(num_epochs):

        i=1
        for video in train_list:
            print(f'File {i} / {len(train_list)} - {video}, {num_epochs - epoch} steps remaining')
            X_video, y_video = get_train_data(video, lip_data_dict, lip_data_dir)
            model.fit(X_video, y_video, epochs=1)
            i+=1

        print(f'Validation on {val_file}')
        val_loss, val_acc = evaluate_model(epoch, model, X_val, y_val)
        print(f'val_loss: {round(val_loss,2)} - val_acc: {round(val_acc,2)}%')

        if val_acc>=best_acc:
            print(f"val_acc improved from {best_acc} to {val_acc}, saving model...")
            best_acc = val_acc
            filename="model_best.hdf5"
            model_path = f'{os.getcwd()}/{filename}'    
            model.save(model_path)
        else:
            print(f"val_acc did not improve from {best_acc}")

    return model

def evaluate_model(epoch,  model, x_test, y_test):
    file = f"metrics.txt"
    
    file_path = f'{os.getcwd()}/{file}' 
    x_test = x_test.astype('float16')
    y_pred = model.predict(x_test)
    #top_acc = []
    for y_p, y in zip(y_pred, y_test): 
        acc = {'top1_acc':0.0, 'top3_acc':0.0, 'top5_acc':0.0}     
        top1 = inverse_one_hot_transform(y_p, k=1)
        top3 = inverse_one_hot_transform(y_p, k=3)
        top5 = inverse_one_hot_transform(y_p, k=5)

        for y_t, tp1 in zip(y, top1):
            print(int_phoneme[y_t[0]], " \t", tp1)
            if int_phoneme[y_t[0]] in tp1:
                acc['top1_acc']+=1
        for y_t, tp3 in zip(y, top3):
            if int_phoneme[y_t[0]] in tp3:
                acc['top3_acc']+=1
        for y_t, tp5 in zip(y, top5):
            if int_phoneme[y_t[0]] in tp5:
                acc['top5_acc']+=1

        y_len = len(top1)
        
        acc['top1_acc']/=y_len
        acc['top3_acc']/=y_len
        acc['top5_acc']/=y_len

        acc['top1_acc'] = round(acc['top1_acc']*100,2)
        acc['top3_acc'] = round(acc['top3_acc']*100,2)
        acc['top5_acc'] = round(acc['top5_acc']*100,2)

        with open(file_path, 'a+') as f:
            f.write(f"{epoch} \t Top-1 accuracy: {acc['top1_acc']}% \t Top-3 accuracy: {acc['top3_acc']}% \t Top-5 accuracy: {acc['top5_acc']}%")
            f.write("\n")
        print(f"Top-1 accuracy: {acc['top1_acc']}%")
        print(f"Top-3 accuracy: {acc['top3_acc']}%")
        print(f"Top-5 accuracy: {acc['top5_acc']}%")
        print("\n")
        return 1-acc['top1_acc'], acc['top1_acc']

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
        val_list += test_val_list[half:]
    
    spkr_type = 'volunteers'
    #print(f"Collecting {spkr_type} data..")
    vol_data_dict, vol_video_list, vol_data_dir = collect_video_files('data', 'volunteers', video_path)
        

    # if spkr_type == 'volunteers':
    #     for vol in trainVolunteerList:      
    #         train_list+=[k for k in vol_video_list if vol in k][2:]      
    for vol in testVolunteerList:
        test_val_list=[k for k in vol_video_list if vol in k][-2:]
        half = len(test_val_list)//2
        #test_list += test_val_list[:half]
        #val_list += test_val_list[half:]
    
    TRAIN = 1
    if TRAIN == 1:
        print(f"Training Model...")
        model = train_model(val_list, train_list, lip_data_dict, lip_data_dir)
        print(f"Training Completed.\n")  

    #filename="model_best.hdf5"
    #filename = "video_model.best.hdf5"
    filename = "model_best.hdf5"
    model_path = f'{os.getcwd()}/{filename}' 
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
        for file in test_list:
            X_test, y_test, video = get_test_data(file, lip_data_dict, lip_data_dir)
            print(f"Evaluating Model on {video}...")
            evaluate_model('final',  model, X_test, y_test)
    except:
        print(f"Model not found. Please train the model first")

