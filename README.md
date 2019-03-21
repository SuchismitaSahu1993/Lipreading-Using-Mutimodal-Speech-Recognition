# Project "Read My Lips":

## Description

In Real World, lipreading is the skill of recognizing what is being said from visual information. It is a challenging problem because of the ambiguity at the word and phoneme level that visual information contains. Multiple sounds share the same shape. For example visemes - different phonemes that produce the same lip sequence  (e .g 'ah' and  'aa', 'p' and 'b'). Visual lipreading plays an important role in human-computer interaction in noisy environments where audio speech recognition is difficult. It can also be extremely useful as a means of communication for the hearing-impaired. In addition, there can be several potential applications like dictating messages in noisy environment and better dealing with multiple simultaneous speakers. 
In recent years, Machine Learning has been more effective than professional lip readers in discerning speeches from the silent video clips.

The idea behind this project is to make use of visual information of the stance of the mouth or lip region in addition to the corresponding audio to predict spoken phonemes. Our work is inspired by the state of the art lipreading algorithm called LIPNET, developed by researchers at the University of Oxford which makes use of spatiotemporal convolutions, a recurrent network, and the connectionist temporal classification loss, trained entirely end-to-end to give sentence-level lipreading.

We implement a similar multimodal system which makes use of video as well as audio features to improve the performance of the lipreading system on a phoneme level. The system is trained on a sequence of images passing through several layers of CNN-LSTM network in conjunction with an audio LSTM network ultimately retrieving the final sequence of phoneme sequence spoken. The model is based on the recent sequence-to-sequence (encoder-decoder with attention) translater architectures that have been developed for speech recognition and machine translation. The model has a dual attention mechanism that can operate over visual sub-network input only, audio-subnetwork input only, or both by learning a language model. CNN is used in conjunction with LSTMs for extracting temporal information.

## Description of Code

* downloadTCDTIMIT - This folder contains the code for downloading the data from TCDTIMIT
* DataPreprocessing - This folder contains the code for preprocessing the dataset.
* audio_networks - This folder contain the code for audio sub-network.
* LipReading - This folder contain the code for video sub-network
* CombinedSR - This folder contains the code for network making predictions combining both audio and video features

## Results

Train Set : Lip Speakers, Test Set : Lip Speakers

* Audio Sub-network : 70.51%
* Video Sub-network : 37.50%
* Combined Network : 41.12%

Train Set : Lip Speakers, Test Set : Volunteers

* Audio Sub-network : 53.20%
* Video Sub-network : 25.67%
* Combined Network : 31.95%

## Model Evaluation:
* The model can be evaluated using EvaluateModelDemo.py script under combinedSR. It needs model file as input. On running, It starts by recording video of person speaking a sentence, cuts it into frames focused on speakers' lips,  passing them to the model and outputs phoneme predictions