#!/usr/bin/python

#### help functions
#from __future__ import print_function

# remove without complaining
import os, os.path
import subprocess
import traceback
import sys
import glob
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.io as sio
import dlib
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from PIL import Image
import scipy.misc
#from fixTCDTIMITwavStructure import *
#from getPhnFiles import *

########### Small helpfunctions ###########

## http://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input#3041990
# query_yes_no("Is cabbage yummier than cauliflower?", None)
# Is cabbage yummier than cauliflower? [y/n] [ENTER]
# Please respond with 'yes' or 'no' (or 'y' or 'n').
# Is cabbage yummier than cauliflower? [y/n] y
# >>> True
def query_yes_no (question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def silentremove (filename):
    try:
        os.remove(filename)
    except OSError as e:  # name the Exception `e`
        # print("Failed with:", e.strerror)  # look what it says
        pass

# sort based on filenames, numerically instead of lexicographically
def tryint (s):
    t = os.path.splitext(s)[0]
    try:
        return int(t)
    except:
        try:
            u = t.split("_")[1]
            return int(u)
        except:
            return t

def writeToTxt(lines, path):
    if (not os.path.exists(os.path.dirname(path))):
        os.makedirs(os.path.dirname(path))
    file = open(path, 'w')
    for line in lines:
        writeLine(lines, line, file)
    file.close()

def writeLine(lines, line, file):
    # don't write newline after last element
    if (lines.index(line) < len(lines) - 1):
        writeNewLine = True
    else:
        writeNewLine = False

    # if a line is a list, there are several elements you want to write on that line. need to convert to str before writing
    if isinstance(line, list) or isinstance(line, tuple):
        line = ' '.join(map(str, line))

    if writeNewLine:
        file.write("%s\n" % line)
    else:
        file.write("%s" % line)
#######################################################
################# Actual functions ####################
#######################################################

# read all the times from the mlf file, split on line with new video
# create a list of lists. The higher-level list contains the block of one video, the 2nd-level list contains all the lines of that video
# MLFfile
#    | video1
#        | firstPhoneme
#        | secondPhoneme
#    | video2
#    etc
# http://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list-with-python#3277516

def readMLFfile (filename):
    with open(filename, "r") as ins:
        array = [[]]
        video_index = 0
        for line in ins:
            line = line.strip('\n')  # strip newlines
            if len(line) > 1:  # don't save the dots lines
                if ".mp4" not in line:
                    array[video_index].append(line)
                else:  # create new 2nd-level list, now store there
                    array.append([line])
                    video_index += 1
    
    return array[1:]


# outputs a list of times where the video should be converted to an image, with the corresponding phonemes spoken at those times
# videoPhonemeList:        list of lines from read file that cover the phonemes of one video,
#                              as well as  a list of the phonemes at those times (for keeping track of which image belongs to what)
# timeModifier:            extract image at beginning, middle, end of phoneme interval?
#                            (value between 0 and 1, 0 meaning at beginning)
def processVideoFile (videoPhonemeList, timeModifier=0.5):
    videoPath = str(videoPhonemeList[0]).replace('"', '')
    videoPath = videoPath.replace('rec', 'mp4')
    phonemes = []  # list of tuples, 1= time. 2=phoneme
    
    for idx, line in enumerate(
            videoPhonemeList[1:]):  # skip the path line; then just three columns with tabs in between
        splittedLine = line.split()  # split on whitespaces
        
        phoneme = splittedLine[2]
        
        start = float(splittedLine[0]) / 10000000
        end = float(splittedLine[1]) / 10000000
        
        # if (idx == 0 ):  #beginning or end = silence, take the best part  #TODO
        #     extractionTime = start
        # elif (idx == len(videoPhonemeList[1:])-1):
        #     extractionTime = end
        # else:
        extractionTime = start * (1 - timeModifier) + (timeModifier) * end
        extractionTime = "{0:.3f}".format(extractionTime)  # three decimals
        
        phonemes.append((extractionTime, phoneme))  # add the (time,phoneme) tuple
    
    return videoPath, phonemes


def processVideoPhonemes(videoPhonemeList, timeModifier=0.5):
    videoPath = str(videoPhonemeList[0]).replace('"', '')
    videoPath = videoPath.replace('rec', 'mp4')

    phonemes = []  # list of tuples, 1= time. 2=phoneme

    for idx, line in enumerate(
            videoPhonemeList[1:]):  # skip the path line; then just three columns with tabs in between
        splittedLine = line.split()  # split on whitespaces

        phoneme = splittedLine[2]

        start = float(splittedLine[0]) / 10000000
        end = float(splittedLine[1]) / 10000000

        # if (idx == 0 ):  #beginning or end = silence, take the best part  #TODO
        #     extractionTime = start
        # elif (idx == len(videoPhonemeList[1:])-1):
        #     extractionTime = end
        # else:
        extractionTime = start * (1 - timeModifier) + (timeModifier) * end
        extractionTime = "{0:.3f}".format(extractionTime)  # three decimals

        phonemes.append((start, end, phoneme, extractionTime))  # add the (time,phoneme) tuple

    return videoPath, phonemes


# get valid times, phonemes, frame numbers
def getValid(time_phonemes, framerate):  # frameRate = 29.97 for the TCDTimit database
    import math
    # take care of duplicates: loop through the phonemes, if two are same frame, only keep the first one
    seen_framePhonemes = set()
    validFrames = []
    validPhonemes = []
    validTimes = []
    for time_phoneme in time_phonemes:
        time = float(time_phoneme[0])
        frame = int(math.floor(time * framerate))
        phoneme = time_phoneme[1]
        if (frame, phoneme) not in seen_framePhonemes:
            validPhonemes.append(time_phoneme[1])
            validTimes.append(time)
            validFrames.append(frame)
            seen_framePhonemes.add(frame)
        else:
            print("frame_phoneme ", (frame, phoneme), " already seen")
    return validTimes, validFrames, validPhonemes


# write file with phonemes and corresponding frame numbers. First column = frames. Second column = corresponding phonemes
def writePhonemesToFile(videoName, speakerName, phonemes, targetDir):
    validTimes, validFrames, validPhonemes = getValid(phonemes, 29.97)
    phonemeFile = ''.join([targetDir, os.sep, speakerName, "_", videoName, "_PHN.txt"])
    if not os.path.exists(targetDir): os.makedirs(targetDir)

    # add 1 to the validFrames to fix the ffmpeg issue (starts at 1 instead of 0)
    for i in range(0, len(validFrames)):
        validFrames[i] += 1
        if validFrames[i] < 1: validFrames[i] = 1

    # check that no frames are larger than last frame extracted by extractAllFrames
    highest = 1
    for root, dirs, files in os.walk(targetDir):
        for file in files:
            #print(file)
            name, ext = os.path.splitext(file)
            if not ext == ".jpg": continue
            frame = int(name.split("_")[1])
            if frame > highest: highest = frame


    for i in range(len(validFrames)):
        if validFrames[i] > highest:
            #print("FOUND HIGHER THAN HIGHEST:", validFrames[i])
            validFrames[i] = highest


    # write to text file
    thefile = open(phonemeFile, 'w')
    for i in range(len(validFrames) - 1):
        item = (validFrames[i], validPhonemes[i])
        thefile.write(' '.join(map(str, item)) + "\r\n")
    item = (validFrames[-1], validPhonemes[-1])
    thefile.write(' '.join(map(str, item)))
    thefile.close()

    # also write a mat file
    matPath = targetDir + os.sep + "phonemeFrames.mat"
    sio.savemat(matPath, {'validFrames': np.array(validFrames), 'validPhonemes': np.array(validPhonemes)})

    return 0

# used to keep folder structure, but under different root path
def fixStoreDirName (storageLocation, videoName, pathLine):
    """
    Fix the path of the root dir of all the newly generated files for this video.
    Gets base path from the MLF file; removes everything from 'Clips' on; adds storeDirName
    For example: file for lipspeaker will be '/home/user/TCDTIMIT/processed/lipspeakers/Lipspkr1'
    :param storeDir: the name of the root dir (which will be just under the 'TCDTIMIT' dir)
    :return:
    """
    storeDir = str(pathLine).replace('"', '')
    storeDir = storeDir.replace('.rec', '.mp4')
    print(storeDir)
    if not "TCDTIMITDataset/" in storeDir: raise Exception("You have to create a 'TCDTIMIT' top level directory!!"); sys.exit(-1)
    oldStoragePath, relPath = storeDir.split("TCDTIMITDataset/")  # /home/data/TCDTIMIT/volunteers/...
    # make sure we're at the volunteers/lipspeakers top dir
    # keep going down until you get to 'lipspeakers' or 'volunteers'
    relTopDir = relPath.split('/')[0]
    while not (relTopDir == 'lipspeakers' or relTopDir == 'volunteers'):
        relPath = '/'.join(relPath.split('/')[1:] )
        relTopDir = relPath.split('/')[0]

    storeDir = ''.join([storageLocation, os.sep, relPath])
    storeDir, second = storeDir.split("Clips")
    if storeDir.endswith('/'):
        storeDir = storeDir[:-1]
    
    # now add the video Name
    storeDir = ''.join([storeDir, os.sep, videoName])
    return storeDir


def deleteUnneededFiles(videoDir):
    #print("deleting files...")
    # read correct frames: firs column of text file
    parentName = os.path.basename(os.path.dirname(videoDir))
    dirName = os.path.basename(videoDir)
    validFrames = []
    with open(videoDir + os.sep + parentName + "_" + dirName + "_PHN.txt") as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                validFrames.append(parts[0])  # print column 2
  

    # walk through the files, if a file doesn't contain '_validFrame', then remove it.
    nbRemoved = 0
    for root, dirs, files in os.walk(videoDir):
        #files.sort(key=tryint)
        for f in files:
            name, ext = os.path.splitext(f)
            filePath = os.path.join(root, f)
            if ext != ".jpg": 
                continue
            fname = os.path.splitext(f)[0]
            fnumber = fname.split("_")[1]
            if fnumber not in validFrames:
                os.remove(filePath)
                nbRemoved += 1

    return nbRemoved


def extractAllFrames (videoPath, videoName, storeDir, framerate, targetSize, cropStartPixel):
    """
    extract all frames from a video, and store them in storeDir
    """
    # 1. [extracting all frames](https://ubuntuforums.org/showthread.php?t=1141293): `mkdir videoName; ffmpeg -i VideoName.mp4 frames/%d.jpg`
    if not os.path.exists(storeDir):
        #pdb.set_trace()# skip already existing videos (it is assumed the exist if the directory exists)
        os.makedirs(storeDir)
        # eg vid1_. frame number and extension will be added by ffmpeg
        outputPath = ''.join(
                [storeDir, os.sep, videoName, "_", ])  # eg .../sa1_3.jpg (frame and extension added by ffmpg)
        
        command = ['ffmpeg',
                   '-i', videoPath,
                   '-s', targetSize,
                   '-vf', "crop=" + targetSize + ":" + cropStartPixel,
                   outputPath + "%d.jpg"]  # add frame number and extension
        
        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE
        subprocess.Popen.wait(p)

        return 1
    
    else: #assume already done
        return 0


# detect faces in all jpg's in sourceDir
# extract faces to "storeDir/faces", and mouths to "storeDir/mouths"


def extractFacesMouths (sourceDir, storeDir, detector, predictor, saveFaces = True, saveMouths = True):

    storeFaceDir = storeDir + os.sep + "faces"
    if not os.path.exists(storeFaceDir):
        os.makedirs(storeFaceDir)
    
    storeMouthsDir = storeDir + os.sep + "mouths"
    if not os.path.exists(storeMouthsDir):
        os.makedirs(storeMouthsDir)

    for f in glob.glob(os.path.join(sourceDir, "*.jpg")):

        dets = []
        fname, ext = os.path.splitext(os.path.basename(f))
        if ext == ".jpg":
            try:
                # print(f)
                facePath = storeFaceDir + os.sep + fname + "_face.jpg"
                mouthPath = storeMouthsDir + os.sep + fname + "_mouth.jpg"
                
                if os.path.exists(facePath):
                    #print(facePath, " already exists")
                    continue

                # load the image, as grayscale and uint8
                img = scipy.misc.imread(f, 'L').astype('uint8')
                width, height = img.shape

                # detect face, then keypoints. Store face and mouth
                # resize with factor 4 to increase detection speed
                resizer = 16
                dim =(int(width / resizer), int(height / resizer))
                imgSmall = scipy.misc.imresize(img, dim)

                dets = detector(imgSmall, 1)  # detect face
                if len(dets) == 0:
                    print("looking on 1/4 image...")
                    resizer = 4
                    dim = (int(width / resizer), int(height / resizer))
                    imgSmall = scipy.misc.imresize(img, dim)

                    dets = detector(imgSmall, 1)

                    if len(dets) == 0:
                        print("looking on full-res image...")
                        resizer = 1
                        dim = (int(width / resizer), int(height / resizer))
                        imgSmall = scipy.misc.imresize(img, dim)

                        dets = detector(imgSmall, 1)
                        if len(dets) == 0:
                            print("still no faces found. Using previous face coordinates...")
                            # if 'top' in locals(): #could be issue if no face in first image ? #TODO
                            try:
                                face_img = img[top:bot, left:right]
                                mouth_img = img[my:my + mh, mx:mx + mw]

                            except: # vars not yet defined b/c this is the first image -> use some manual default values
                                top= int(height*0.2); bot=int(height*0.9)
                                left = int(width*0.2); right = int(width*0.8)
                                face_img = img[top:bot, left:right]

                                my = int(height * 0.6);   mh = int(height * 0.3)
                                mx = int(width * 0.2);    mw = int(width * 0.5)
                                mouth_img = img[my:my + mh, mx:mx + mw]

                            if saveFaces: scipy.misc.imsave(facePath, face_img)  # save face image
                            if saveMouths: scipy.misc.imsave(mouthPath, mouth_img)
                            continue

                d = dets[0]
                if saveFaces:
                    # extract face, store in storeFacesDir
                    left = d.left() * resizer
                    right = d.right() * resizer
                    top = d.top() * resizer
                    bot = d.bottom() * resizer
                    # go no further than img borders
                    if (left < 0):      left = 0
                    if (right > width): right = width
                    if (top < 0):       top = 0
                    if (bot > height):  bot = height
                    face_img = img[top:bot, left:right] #img.crop((left,top,right,bot))#
                    scipy.misc.imsave(facePath, face_img)  # save face image

                if saveMouths:# now detect mouth landmarks
                    # detect 68 keypoints, see dlibLandmarks.png
                    shape = predictor(imgSmall, d)
                    # Get the mouth landmarks.
                    mx = shape.part(48).x * resizer
                    mw = shape.part(54).x * resizer - mx
                    my = shape.part(31).y * resizer
                    mh = shape.part(57).y * resizer - my
                    # go no further than img borders
                    if (mx < 0):       mx = 0
                    if (mw > width):   mw = width
                    if (my < 0):       my = 0
                    if (mh > height):  mh = height

                    # scale them to get a better image of the mouth
                    # default : width 1.5, height 1
                    widthScalar = 1.25
                    heightScalar = 0.85
                    mx = int(mx - (widthScalar - 1) / 2.0 * mw)
                    # my = int(my - (heightScalar - 1)/2.0*mh) #not needed, we already have enough nose
                    mw = int(mw * widthScalar)
                    mh = int(mh * widthScalar)

                    mouth_img = img[my:my + mh, mx:mx + mw] #img.crop((mx,my,mx+mw,my+mh))
                    scipy.misc.imsave(mouthPath, mouth_img)

            except:
                print("Unexpected error:", sys.exc_info()[0])
                print(traceback.format_exc())
                raise


def resize_image (filePath, filePathResized, keepAR=True, width=120.0):
    im = io.imread(filePath)
    if keepAR: #Aspect Ratio
        r = width / im.shape[1]
        dim = (int(im.shape[0] * r), int(width))
        im_resized = resize(im, dim)
    else:
        im_resized = resize(im, (120, 120))
    io.imsave(filePathResized, im_resized)


def resizeImages (rootDir, dirNames, keepAR=True, width=640.0):
    for dirName in dirNames:
        dirPath = rootDir + os.sep + dirName
        targetDirPath = rootDir + os.sep + dirName + "_" + str(int(width))
        if not os.path.exists(targetDirPath):
            os.makedirs(targetDirPath)
        
        # loop through the files
        onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
        onlyfiles.sort(key=tryint)
        for file in onlyfiles:
            filename, ext = os.path.splitext(file)
            filePath = ''.join([dirPath, os.sep, file])  # the file we're processing now
            targetPath = ''.join([targetDirPath, os.sep, file])
            # print("resizing ", filePath, " to ", targetPath)
            if not os.path.exists(targetPath):
                resize_image(filePath, targetPath, keepAR, width)
    return 0


# convert all images in folders specified in the list 'dirNames', that are found under the rootDir, to grayscale.
def convertToGrayScale (rootDir, dirNames):
    nbConverted = 0
    for root, dirs, files in os.walk(rootDir):
        files.sort(key=tryint)
        for file in files:
            parentDir = os.path.basename(root)
            fname = os.path.splitext(file)[0]  # no path, no extension. only filename
            if parentDir in dirNames:
                # convert all images in here to grayscale, store to dirName_gray
                newDirPath = ''.join([os.path.dirname(root), os.sep, parentDir + "_gray"])
                newFilePath = ''.join([newDirPath, os.sep, fname + "_gray.jpg"])
                if not os.path.exists(newDirPath):
                    os.makedirs(newDirPath)
                if not os.path.exists(newFilePath):
                    # read in grayscale, write to new path
                    # with OpenCV: weird results (gray image larger than color ?!?)
                    # img = cv2.imread(root+os.sep+file, 0)
                    # cv2.imwrite(newFilePath, img)
                    
                    img_gray = rgb2gray(io.imread(root + os.sep + file))
                    io.imsave(newFilePath, img_gray)  # don't write to disk if already exists
                    nbConverted += 1
    
    # print(nbConverted, " files have been converted to Grayscale")
    return 0