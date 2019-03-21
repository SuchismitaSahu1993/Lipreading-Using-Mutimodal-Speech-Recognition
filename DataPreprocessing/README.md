Hi there.  
This file explains how to use the scripts in this folder in order to process the TCDTIMIT dataset.

The goal is to extract from mp4 videos and mlf phoneme label files the video frames that correspond to the phonemes in the label file.

What you'll need:  
- python (3.6 or up). I recommend using Anaconda
- dlib (image processing library). Anaconda installation command: 'conda install -c menpo dlib'
- skimage. Anaconda installation command: 'conda install -c  Anaconda scikit-image'
- tqdm. Anaconda installation command: 'conda install tqdm'   

The most important folders and files are:
1. in the folder MLFfiles: `lipspeaker_labelfiles.mlf` and `volunteer_labelfiles.mlf`:
    - these files are used as input in the main.py file. 
    - You can use the `lipspeaker_test.mlf` for a (much) smaller version of the database, to try it out. You'll have to extract the zip files first, though. I would suggest to use`volunteer_labelfiles_sample.mlf` for initial try out.

2. `main.py`: 
    - use this to specify the main MLF file, which contains paths of the videos to be processed, and start-and stop times for each phoneme in the videos.  You also set the target directory and the number of threads to use.  
    - In order to process the database, you need to change all the paths in this MLF file to the paths where you downloaded the database.  
            
    1. `processDatabase_v2.py`:
          - this file contains the main functions that process the downloaded database. It works multithreadedly.  
          - extracted images and files will be saved to storageLocation (set in main.py i.e outside repo in a folder named TIMIT), but further the directory structure of the database will be kept (so an extracted frame is stored in storageLocation/lipreaders/Lipspkr1/sa1/extractedFrame.jpg)  
          - it works in several steps:    
               1. preprocess the MLF file: extract video paths, phonemes and their times, and the corresponding frames in the video.    
               1. extract all frames from the video, cropping the video to some box roughly corresponding to the face (to make further processing easier)
               1. write phoneme-frame information to both a txt and a mat file
               1. remove unneccessary frames
               1. extract faces and mouths, storing them in 'faces' and 'mouths' folders. Also converts to grayscale.  
               1. resize the mouth images, to make them all the same size
          - there are several parameters you can change for your application
    1. `helpFunctions.py`: contains the implementations of functions that are used in processDatabase.py

1. `fileDirOps.py`:  This is used to filter the extracted database files for grayscale resized mouth images and the phoneme-frame txt files. The phoneme corresponding to each image was added to the image name. 
The files were then copied to another directory, and pickled for later usage in neural networks.

1. other files:
    1. `EGillenThesis.pdf`:   the masters thesis of Eoin Gillen, Trinity College Dublin, who helped create the database. This contains lots of info about how the database was made.    
    1. `phonemeList.txt`:     contains a list of all possible phonemes (39). This is somewhat simplified from the full english phoneme set, see EGillenThesis.pdf. These are also in CMU_..._phones.txt  
    1. `CMU_Pronunciation_Dictionary`: can be used to map phonemes to words or the other way around. See for example `https://github.com/jimkang/word-phoneme-map`  
    1. `countNbPhonemes.py`:  counts the number of times the database contains each phoneme. Useful for estimating how well you'll be able to train.  
    1. `dlib.so`:            needed to be in this folder in order for the face and mouth extraction to work.  
    1. `shape_predictor_68_face_landmarks.dat`: used by dlib to detect 68 facial landmarks, wihch are used to the mouth region. See `dlibLandmarks.png` for a visualization of the landmark locations.  
    
    
In short: in order to use the database, do this:
1. update the MLF files for lipspeakers and volunteers to point to wherever you downloaded and extracted the database
2. change the paths in main.py, i suggest first start with sample volunteer MLFs. Execute it.
3. wait till done
4. you can use extractTCDTIMITaudio on the processed directory to get the wavs and phn files. Further processing -> see [audioSR](https://github.com/matthijsvk/multimodalSR/tree/master/code/audioSR)  
