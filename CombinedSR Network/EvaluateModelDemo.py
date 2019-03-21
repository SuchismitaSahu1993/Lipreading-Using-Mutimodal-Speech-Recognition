import os
import json
import pdb
import warnings
import dlib
import itertools
import librosa
import sys
sys.path.append('..')
from PIL import Image
import numpy as np
import keras.backend as K
from queue import PriorityQueue
from keras.models import Model, Sequential, load_model
from phonemeSet import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from DataPreprocessing.helpFunctions import *
from demo import *

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
    elif resp == 'Demo':
        speakerDir = root_dir+'/Demo'

    data_dict = {}
    fileName = f'{data_type}_dict.json'
    if os.path.exists(f'{os.getcwd()}/{fileName}'):    
        data = open(speakerDir+os.sep+fileName).read()
        data_dict = json.loads(data)

    video_list = []
    speaker  = f'{speakerDir}/data_phn.txt'
    if os.path.exists(f'{os.getcwd()}/{speaker}'):
        with open(speaker, 'r') as f:
            lines = f.readlines()
        video_list = [l.strip() for l in lines]
    return data_dict, video_list, speakerDir

def get_predict_data(data_dir):
    data_files = []
    for root, dirs, files in os.walk(data_dir):
        data_files = files
    time_steps = len(data_files)    
    X = np.zeros((time_steps,120,120,1))
    for i, ID in enumerate(data_files):
        im = np.array(Image.open(data_dir+'/'+ID), dtype=np.uint8)
        im = np.expand_dims(im,3)
        im = im.astype('float32')/255
        X[i] = im
    x_test = X.reshape(1, X.shape[0], 120, 120, 1)
    return x_test

def evaluate_model(epoch,  model, x_test):
    x_test = x_test.astype('float16')
    y_pred = model.predict(x_test)

    top1 = inverse_one_hot_transform(y_pred[0], k=1)
    #top_acc = []
    print(f"Phoneme Predictions:")
    print(top1)

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
    while k + 9 < phon_split_audio.shape[0]:
        temp_x = phon_split_audio[k:k+9]
        X.append(np.array(temp_x))
        k += 1
    return X

def create_audio_test_dataset(filename):
    phon_split_audio = []
    audio, orig_sr = librosa.load(filename)
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
    mfcc = librosa.feature.mfcc(audio, sr=16000, n_mfcc=13).T 
    phon_split_audio.extend(mfcc)
    phon_split_audio = np.array(phon_split_audio)

    return phon_split_audio

if __name__=="__main__":

    # audio_data = create_audio_test_dataset('test.wav')
    # audio_test = preprocessing_for_rnn(audio_data, stack_size=9)
    # audio_test = np.array(audio_test)

    # model = load_model('best_model_more_volunteers.hdf5')
    # audio_pred = model.predict(audio_test)
    # new_audio_pred = []
    # for pred in audio_pred:
    #     new_audio_pred.append(list(pred[5]))
    # new_audio_pred = np.vstack(new_audio_pred)
    # pred_one = inverse_one_hot_transform(new_audio_pred)
    # print(pred_one)
    print(f"Recording video...")

    capture_video()
    #extract_audio()
    extract_frames()

    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(predictor_path):
        print('Landmark predictor not found!')

    predictor = dlib.shape_predictor(predictor_path)
    videoPath = './'
    saveFaces = True
    saveMouths = True
    #  extract faces and mouths
    tick = time.clock()
    storeDir = "faces and mouths"
    sourceDir = storeDir
    print("Extracting mouths... ", sourceDir)
    extractFacesMouths('test_frames', storeDir, detector, predictor, saveFaces, saveMouths)
    print("\tAll faces and mouths have been extracted.")
    print("duration: ", time.clock() - tick)
    print("----------------------------------")

    #  resize mouth images, for convnet usage: 120x120
    tick = time.clock()
    dirNames = []
    if saveMouths: dirNames.append("mouths")
    if saveFaces: dirNames.append("faces")  # dirNames = ["mouths_gray", "faces_gray"]
    print("Resizing images...")
    resizeImages(storeDir, dirNames, False, 120.0)
    print("\tAll mouths have been resized.")
    print("duration: ", time.clock() - tick)
    print("----------------------------------")

    #video_path = os.path.join(os.path.expanduser('../../TCDTIMIT/video'))
    video_dir = './faces and mouths/mouths_120'

    #extract_phoneme(word_list)
        
    filename = "video_model_best`.hdf5"
    model_path = f'{os.getcwd()}/{filename}' 
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            X_test = get_predict_data(video_dir)
            print(f"Evaluating Model...")
            evaluate_model('final',  model, X_test)
    except:
        print(f"Model not found. Please train the model first")

    # sentence=input("Enter the sentence to be spoken:")
    # extract_phoneme(sentence)