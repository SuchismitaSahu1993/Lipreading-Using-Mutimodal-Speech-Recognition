from __future__ import print_function

# this file contains different operations on files and directories:
#   1. fixNames: files generated with old functions give mouth images, stored as 'videoName_faces_frameNb.jpg'
#                Transform this to the format 'videoName_frameNb_faces.jpg'
#

import getopt
import zipfile, os.path
import concurrent.futures
import glob
import threading
import collections
from collections import OrderedDict
import os, errno
import sys
sys.path.append('..')
import subprocess
import shutil
from utils.helpFunctions import *
from phoneme_set import *
from shutil import copyfile
import numpy as np
from PIL import Image
import pickle
import time
import json


# 1. remove all specified directories and their contents
# a rootdir, and a list of dirnames to be removed
# THIS FUNCTION deletes all specified directories AND their contents !!!
# Be careful!

def deleteDirs(rootDir, names):
    dirList= []
    for root, dirs, files in os.walk(rootDir):
        for dirname in dirs:
            for name in names:
                if name in dirname:
                    path = ''.join([root, os.sep, dirname])
                    dirList.append(path)
    print(dirList)
    if query_yes_no(f"Are you sure you want to delete all these directories AND THEIR CONTENTS under {rootDir}?" , "yes"):
        nbRemoved = 0
        for dir in dirList:
            print(f'Deleting dir: {dir}')
            shutil.rmtree(dir)
            nbRemoved +=1
        print(f"{nbRemoved} directories have been deleted")
    return dirList

# stuff for getting relative paths between two directories
def pathsplit(p, rest=[]):
    (h,t) = os.path.split(p)
    if len(h) < 1: return [t]+rest
    if len(t) < 1: return [h]+rest
    return pathsplit(h,[t]+rest)

def commonpath(l1, l2, common=[]):
    if len(l1) < 1: return (common, l1, l2)
    if len(l2) < 1: return (common, l1, l2)
    if l1[0] != l2[0]: return (common, l1, l2)
    return commonpath(l1[1:], l2[1:], common+[l1[0]])

# p1 = main path, p2= the one you want to get the relative path of
def relpath(p1, p2):
    (common,l1,l2) = commonpath(pathsplit(p1), pathsplit(p2))
    p = []
    if len(l1) > 0:
        p = [ '../' * len(l1) ]
    p = p + l2
    #p.append(l2[0])
    return os.path.join( *p )

def newrelpath(p1, p2):
    (common,l1,l2) = commonpath(pathsplit(p1), pathsplit(p2))
    p = []
    if len(l1) > 0:
        p = [ '../' * len(l1) ]
    #p = p + l2
    p.append(l2[0])
    return os.path.join( *p )

# 2. copy a dir structure under a new root dir
# copy all mouth files to a new dir, per speaker. Also remove the 'mouths_gray_120' directory, so the files are directly under the videoName folder
# -> processed/lipspeakers

# helpfunction: fix shutil.copytree to allow writing to existing files and directories (http://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth#12514470)
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
            
def copyDBFiles(rootDir, names, targetRoot):
    
    dirList = []
    for root, dirs, files in os.walk(rootDir):
        for dir in dirs:
            for name in names:
                if name in dir:
                    path = ''.join([root, os.sep, dir])
                    dirList.append(path)
    
    print(f"Directories to be copied: {dirList[0:10]}")

    if query_yes_no(f"Are you sure you want to copy all these directories {rootDir} to {targetRoot}?" , "yes"):
        nbCopiedDirs = 0
        
        for dir in dirList:
            relativePath = newrelpath(rootDir, dir)
            dest = ''.join([targetRoot+relativePath])
            print(f"copying dir: {dir} to {dest}")
            copytree(dir, dest)
            nbCopiedDirs +=1
            
        print(f"{nbCopiedDirs} directories have been copied to {targetRoot}")
    return dirList


# need this to traverse directories, find depth
def directories (root):
    dirList = []
    for path, folders, files in os.walk(root):
        print(path)
        for name in folders:
            dirList.append(os.path.join(path, name))
    return dirList

def depth(path):
    return path.count(os.sep)


# extract phonemes for each image, put them in the image name.
# moveToSpeakerDir: - flattens dir structure: to also copy all the jpg and phn files to the speaker dir (so not 1 dir per video)
#                   - also renames: remove speakername, replace '_PHN.txt' by '.vphn'
def addPhonemesToImageNames(videoDir, moveToSpeakerDir=False):
    print(f"processing: {videoDir}")
    
    # videoDir will be the lowest-level directory
    videoBaseDir = videoDir.replace('/mouths_120','')
    videoName= os.path.basename(videoBaseDir)
    parentName = os.path.basename(os.path.dirname(videoBaseDir))
    validFrames = collections.OrderedDict({})
    phoneme_extension = "_PHN.txt"
    phonemeFileName = parentName + "_" + videoName + phoneme_extension
    phonemeFile = ''.join([videoBaseDir + os.sep + phonemeFileName])
    
    with open(phonemeFile) as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                validFrames[str(parts[0])] = parts[1]  # dict, key= frame, value = phoneme
                
    copyfile(phonemeFile, videoDir+os.sep+phonemeFileName)
    nbRenamed = 0
    for root, dirs, files in os.walk(videoDir):
        for file in files:
            fileName, ext = os.path.splitext(file)
            if ext == ".jpg":
                filePath = ''.join([root, os.sep, file])
                parentName = file.split("_")[0]
                videoName = file.split("_")[1]
                frameNumber = file.split("_")[2]
                if frameNumber in validFrames.keys():
                    phoneme = validFrames[frameNumber]
                    if len(frameNumber) == 2:
                        frameNumber = '0'+frameNumber
                    newFileName = ''.join([parentName, "_", videoName, "_", frameNumber, "_", phoneme, ext])
                    parent = os.path.dirname(root)
                    newFilePath = ''.join([parent, os.sep,'mouths_120',os.sep, newFileName])
                    print(f"{filePath} will be renamed to: newFilePath")
                    os.rename(filePath, newFilePath)
                    nbRenamed += 1
                    
    print(f"Finished renaming {nbRenamed} files.")
    return 0


# now traverse the database tree and rename  files in all the directories
def addPhonemesToImagesDB(rootDir, moveToSpeakerDir=True):
    dirList = []
    
    for dir in directories(rootDir):
        if depth(relpath(rootDir, dir)) == 3 and os.path.basename(dir) == 'mouths_120':
            dirList.append(dir)
            
    print(f"Directories to be processed: {dirList[0:10]}")
    for dir in dirList:
        addPhonemesToImageNames(dir, moveToSpeakerDir=moveToSpeakerDir)
        if moveToSpeakerDir: shutil.rmtree(dir)
    return 0

def save_to_file(labels_dict, path):    
    with open(path+'/image_labels_dict.json', 'w') as f:
        json.dump(labels_dict, f)

# for training on visemes
def getPhonemeToVisemeMap():
    map = {'f':'A','v':'A',
            'er':'B','ow':'B','r':'B','q':'B','w':'B','uh':'B','uw':'B','axr':'B','ux':'B',
             'b':'C','p':'C','m':'C','em':'C',
             'aw':'D',
             ' dh':'E','th':'E',
             'ch':'F','jh':'F','sh':'F','zh':'F',
             'oy':'G', 'ao':'G',
             's':'H', 'z':'H',
             'aa':'I','ae':'I','ah':'I','ay':'I','ey':'I','ih':'I','iy':'I','y':'I','eh':'I','ax-h':'I','ax':'I','ix':'I',
             'd':'J','l':'J','n':'J','t':'J','el':'J','nx':'J','en':'J','dx':'J',
             'g':'K','k':'K','ng':'K','eng':'K',
             'sil':'S','pcl':'S','tcl':'S','kcl':'S','bcl':'S','dcl':'S','gcl':'S','h#':'S','#h':'S','pau':'S','epi':'S'
    }
    return map

# helpfunction
def getPhonemeNumberMap (
        phonemeMap="../background/phonemeLabelConversion.txt"):
    phonemeNumberMap = {}
    with open(phonemeMap) as inf:
        for line in inf:
            parts = line.split()    # split line into parts
            if len(parts) > 1:      # if at least 2 parts/columns
                phonemeNumberMap[str(parts[0])] = parts[1]  # part0= frame, part1 = phoneme
                phonemeNumberMap[str(parts[1])] = parts[0]
    return phonemeNumberMap

# helpfunction
def getVisemeNumberMap (
        visemeMap="../background/visemeLabelConversion.txt"):
    visemeNumberMap = {}
    with open(visemeMap) as inf: 
        for line in inf:
            parts = line.split()    # split line into parts
            if len(parts) > 1:      # if at least 2 parts/columns
                visemeNumberMap[str(parts[0])] = parts[1]  # part0= frame, part1 = viseme
                visemeNumberMap[str(parts[1])] = parts[0]
    return visemeNumberMap

def speakerToDict(speakerDir):
    labels = OrderedDict({})
    
    for root, dirs, files in os.walk(speakerDir):    
        for file in files:
            name, extension = os.path.splitext(file)
            if extension == ".jpg":
                speakerName, videoName, frame, phoneme = name.split("_")
                phoneme = phoneme.split('.')[0]
                phonemeNumberMap = phoneme_set_39
                labelNumber = phonemeNumberMap[phoneme]
                labels[file]= labelNumber

    return labels

def allSpeakersToDict(databaseDir):
    rootDir = databaseDir
    dirList = []
    labels_dict = OrderedDict({})
    for dir in directories(rootDir):
        if 'processed' not in dir:
            dirList.append(dir)
            
    for speakerDir in dirList:
        print(f"Dictionary Creation for: {speakerDir}")
        labels = speakerToDict(speakerDir)
        labels_dict.update(labels)
        save_to_file(labels_dict, speakerDir)
    return 0
        
if __name__ == "__main__":

    # use this to copy the grayscale files from 'processDatabase' to another location, and fix their names with phonemes
    # then create dict files useable by lipreading network
    
    processedDir = os.path.expanduser("../../../TCDTIMIT/video/processed")
    databaseDir = os.path.expanduser("../../../TCDTIMIT/video/")
    
    # 1. extract phonemes for each image, put them in the image name
    # has to be called against the 'processed' directory
    print("Adding phoneme to filenames...")
    addPhonemesToImagesDB(processedDir, moveToSpeakerDir=False)
    print("-----------------------------------------")
    
    # 2. copy mouths_gray_120 images and PHN.txt files to targetRoot. Move files up from their mouths_gray_120 dir to the video dir (eg sa1)
    print("Copying mouth_gray_120 directories to database location...")
    copyDBFiles(processedDir, ["mouths_120"], databaseDir)
    print("-----------------------------------------")
    
    # 3. create image label dict 
    print("Creating Image-label dict...")
    allSpeakersToDict(databaseDir)
    print("-----------------------------------------")
    
    
