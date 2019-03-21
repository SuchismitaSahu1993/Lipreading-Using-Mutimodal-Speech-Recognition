import numpy as np

#!/usr/bin/env python

import cv2
import os, os.path
import subprocess
import time
import sys
sys.path.append('..')
from PIL import Image
import nltk
from itertools import product as iterprod
import dlib

def capture_video():
        # Start default camera
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FPS, 30)
    
    #fps = video.get(cv2.CAP_PROP_FPS)
    #print("Frames per second using video.get(cv2.CAP_PROP_FPS):", fps)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30, (640,480))

    # Read until video is completed
    while(True):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            out.write(frame)
            # Display the resulting frame
            cv2.imshow('Frame',frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video
    video.release()
    out.release()
    # Closes all the framesl
    cv2.destroyAllWindows()

def extract_frames():
    # create a folder to store extracted images
    folder = './test_frames'
    # if not os.listdir(folder):
    #     for the_file in os.listdir(folder):
    #         file_path = os.path.join(folder, the_file)
    #         if os.path.isfile(file_path):
    #             print("delete file")
    #             os.unlink(file_path)
        
    # use opencv to do the job
    vidcap = cv2.VideoCapture('output.mp4')
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), image)     # save frame as JPEG file
        count += 1
    print("{} images are extacted in {}.".format(count,folder))

def extract_audio():
    
    import subprocess
    command = "ffmpeg -i output.mp4 -ab 160k -ac 2 -ar 44100 -vn audio.wav"
    subprocess.call(command, shell=True)

def wordbreak(s):
    try:
        arpabet = nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download('cmudict')
        arpabet = nltk.corpus.cmudict.dict()
    s = s.lower()
    if s in arpabet:
        return arpabet[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]
    return None

def extract_phoneme(sentence):
    
    phn_list=[]
    word_list=sentence.split(" ")
    for word in word_list:
        word_phn=wordbreak(word)[0]
        for iphn,phn in enumerate(word_phn):
            word_phn[iphn] = ''.join([i for i in phn if not i.isdigit()])
            phn_list.append(word_phn[iphn])

    for phn in phn_list:
        print (phn)

if __name__ == "__main__":
    extract_phoneme("fly away")


