from processDatabase_v2 import *


# get phone list from file, extract phonemes and times, get the frames corresponding to those phonemes
# then remove frames without phonemes, extract faces, extract mouths, convert them to grayscale images
# also store compressed (eg 120x120) versions of grayscale faces and mouths

###################################################################################################
# !!!! Before running this, make sure all the paths to yuythe videos in the MLF file are correct !!!!#0
###################################################################################################
startTime = time.clock()

nbThreads = 20
#processDatabase('./MLFfiles/lipspeaker_labelfiles.mlf',os.path.expanduser("./TCDTIMIT/lipreading/processed"), nbThreads) #storeDir requires TCDTIMIT in the name

#TODO: uncomment above and also change below volunteer_labelfiles_sample to original file once dummy testing is done
processDatabase('./MLFfiles/volunteer_test.mlf', os.path.expanduser("../../TCDTIMIT/video/processed"), nbThreads)
#processDatabase('./MLFfiles/lipspeaker3_labelfiles.mlf', os.path.expanduser("../../TCDTIMIT/video/processed"), nbThreads)

# processDatabase('./MLFfiles/lipspeaker2_test.mlf', os.path.expanduser("~/TCDTIMIT/lipreading/processedTEST3"),
#                 nbThreads)  # storeDir requires TCDTIMIT in the name


duration = time.clock() - startTime
print("This took ", duration, " seconds")





