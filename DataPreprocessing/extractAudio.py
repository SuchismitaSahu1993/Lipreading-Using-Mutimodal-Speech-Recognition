import os, sys
from tqdm import tqdm
import shutil
from helpFunctions import * #processVideoPhonemes, readMLFfile, writeToTxt

def generatePHN(MLFfile, dstDir):
    videos = readMLFfile(MLFfile)

    for video in tqdm(videos):
        try:
            videoPath, phonemes = processVideoPhonemes(video)  # phonemes contain start and end times in seconds
            phonemes =  timeToFrame(phonemes)
            videoDir = os.path.splitext(videoPath)[0]
            videoName = os.path.basename(videoDir)
            speakerPath = os.path.dirname(os.path.dirname(os.path.dirname(videoDir)))
            #speakerName = os.path.basename(speakerPath)

            if not "TCDTIMITDataset/" in speakerPath: raise Exception(
                "Can't extract phonemes; you have to create a 'TCDTIMIT' top level directory!!"); sys.exit(-1)
            oldStoragePath, relPath = speakerPath.split("TCDTIMITDataset/")  # /home/data/TCDTIMIT/volunteers/...

            # remove unneeded folders; store directly under TCDTIMIT/
            topDir, speakerName = relPath.split('/')
            while not (topDir == "volunteers" or topDir == "lipspeakers"):
                relPath = ''.join(relPath.split('/')[1:])
                topDir = relPath.split('/')[0]
            
            oldaudioPath = ''.join([videoDir, ".wav"])
            storeDir = ''.join([dstDir, os.sep, relPath])
            if storeDir.endswith('/'):  storeDir = storeDir[:-1]
            #storeDir = ''.join([storeDir, os.sep, videoName])
            #storeDir = ''.join([storeDir])
            storeDir = os.path.dirname(storeDir)
            
            phonemePath = ''.join([storeDir, os.sep, speakerName, '_', videoName, ".phn"])
            newaudioPath = ''.join([storeDir, os.sep, speakerName, '_', videoName, ".wav"])
            writeToTxt(phonemes, phonemePath)
            shutil.copy(oldaudioPath, newaudioPath)
        except:
            print(f"File {video} not found")
            # print("Extracting PHN files from ", videoPath, ", saving to: \t", storeDir)
            # print phonemePath

    tcdtimitdir = os.path.dirname(os.path.dirname(speakerPath))
    return tcdtimitdir

def timeToFrame(phonemes):
    phonemeFrames = []
    for pair in phonemes:
        startFrame = int(float(pair[0]) * 16000)  #seconds to audioSamples @ 16kHz
        endFrame = int(float(pair[1]) * 16000)
        phoneme = pair[2]

        phonemeFrames.append( (startFrame, endFrame, phoneme)) #, extractionFrame) )  #TODO add extractionframe if extraction moment is changed for lipreading
    return phonemeFrames


if __name__ == '__main__':
#    MLFfile = sys.argv[1]
#    dstDir = sys.argv[2]
#    generatePHN(MLFfile, dstDir)
    srcDir = "../../TCDTIMITDataset/"
    dstDir = os.path.expanduser("../../TCDTIMIT/audio/")
    
    ## PHN: generate phoneme files
    print("EXTRACTING PHNS and WAVS...")
    
    print("volunteers:")
    generatePHN("./MLFfiles/volunteersTest_labelfiles.mlf", dstDir)
    
#    print("lipspeakers:")
#    generatePHN("./MLFfiles/lipspeaker_labelfiles.mlf", dstDir)

    ## not needed because the files already are in the dstDir
    # copyFilesOfType(tcdtimitdir, dstDir, '.phn')
    
    
    ## Example
    # dstDir = os.path.expanduser("~/TCDTIMIT/TCDTIMITaudio")
    # tcdtimitdir = generatePHN('./MLFfiles/lipspeaker_labelfiles.mlf',dstDir)
    # generatePHN('./MLFfiles/volunteer_labelfiles.mlf',dstDir)

