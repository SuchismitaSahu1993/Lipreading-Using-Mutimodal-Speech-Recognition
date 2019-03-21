# using the 39 phone set proposed in (Lee & Hon, 1989)
# Table 3. Mapping from 61 classes to 39 classes, as proposed by Lee and Hon, (Lee & Hon,
# 1989). The phones in the left column are folded into the labels of the right column. The
# remaining phones are left intact.
import logging
logger_phonemeSet = logging.getLogger('phonemeSet')
logger_phonemeSet.setLevel(logging.ERROR)


############  VISEMES  ##############
#viseme_set_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H','I','J','K','S']
viseme_set_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H','S','V1','V2','V3', 'V4']
values = [i for i in range(0, len(viseme_set_list))]
viseme_set = dict(zip(viseme_set_list, values))
classToViseme = dict((v, k) for k, v in viseme_set.items())

phonemeToViseme39 = {
    'ao': 'V1', 'uw': 'V2', 'ih':'V4', 's':'B', 'zh':'D', 'dh':'F',
    'ah': 'V1', 'uh': 'V2', 'iy':'V4', 'z':'B', 'ch':'D', 'f':'G',
    'aa': 'V1', 'ow': 'V2', 'ax':'V4', 't':'C', 'jh':'D', 'v':'G',
    'er': 'V1', 'ae': 'V3', 'l':'A',   'd':'C', 'p':'E',  'ng':'H',
    'oy': 'V1', 'eh': 'V3', 'el':'A',  'n':'C', 'b':'E',  'g':'H',
    'aw': 'V1', 'ey': 'V3', 'r':'A',   'en':'C', 'm':'E', 'k':'H',
    'hh': 'V1', 'ay': 'V3', 'y':'A',   'sh':'D', 'th':'F', 'w':'H',
    'sil':'S', 'sp':'S'
}
# for training on visemes
phonemeToViseme = {
        'f':   'A', 'v': 'A',
        'er':  'B', 'ow': 'B', 'r': 'B', 'q': 'B', 'w': 'B', 'uh': 'B', 'uw': 'B', 'axr': 'B', 'ux': 'B',
        'b':   'C', 'p': 'C', 'm': 'C', 'em': 'C',
        'aw':  'D',
        'dh': 'E', 'th': 'E',
        'ch':  'F', 'jh': 'F', 'sh': 'F', 'zh': 'F',
        'oy':  'G', 'ao': 'G',
        's':   'H', 'z': 'H',
        'aa':  'I', 'ae': 'I', 'ah': 'I', 'ay': 'I', 'ey': 'I', 'ih': 'I', 'iy': 'I', 'y': 'I', 'eh': 'I', 'ax-h': 'I',
        'ax':  'I', 'ix': 'I',
        'd':   'J', 'l': 'J', 'n': 'J', 't': 'J', 'el': 'J', 'nx': 'J', 'en': 'J', 'dx': 'J',
        'g':   'K', 'k': 'K', 'ng': 'K', 'eng': 'K',
        'sil': 'S', 'pcl': 'S', 'tcl': 'S', 'kcl': 'S', 'bcl': 'S', 'dcl': 'S', 'gcl': 'S', 'h#': 'S', '#h': 'S',
        'pau': 'S', 'epi': 'S'
    }

###########  PHONMEMES   ##############
phoneme_set_61_39 = {
    'ao':   'aa',  # 1
    'ax':   'ah',  # 2
    'ax-h': 'ah',
    'axr':  'er',  # 3
    'hv':   'hh',  # 4
    'ix':   'ih',  # 5
    'el':   'l',  # 6
    'em':   'm',  # 6
    'en':   'n',  # 7
    'nx':   'n',
    'eng':  'ng',  # 8
    'zh':   'sh',  # 9
    "ux":   "uw",  # 10
    "pcl":  "sil",  # 11
    "tcl":  "sil",
    "kcl":  "sil",
    "qcl":  "sil",
    "bcl":  "sil",
    "dcl":  "sil",
    "gcl":  "sil",
    "h#":   "sil",
    "#h":   "sil",
    "pau":  "sil",
    "epi":  "sil",
    "q":    "sil",
}

# from https://www.researchgate.net/publication/275055833_TCD-TIMIT_An_audio-visual_corpus_of_continuous_speech
phoneme_set_39_list = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow',
    'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx',
    'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil'
]
values = [i for i in range(0, len(phoneme_set_39_list))]
phoneme_set_39 = dict(zip(phoneme_set_39_list, values))
classToPhoneme39 = dict((v, k) for k, v in phoneme_set_39.items())

# from http://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database, page 5
phoneme_set_61_list = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr',
    'ax-h', 'jh',
    'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'nx',
    'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi',
    'h#',
]
values = [i for i in range(0, len(phoneme_set_61_list))]
phoneme_set_61 = dict(zip(phoneme_set_61_list, values))

def convertPredictions(predictions, phoneme_list=classToPhoneme39, valid_frames=None, outputType="phonemes"):
    # b is straight conversion to phoneme chars
    predictedPhonemes = [phoneme_list[predictedClass] for predictedClass in predictions]

    # c is reduced set of b: duplicates following each other are removed until only 1 is left
    reducedPhonemes = []
    for j in range(len(predictedPhonemes) - 1):
        if predictedPhonemes[j] != predictedPhonemes[j + 1]:
            reducedPhonemes.append(predictedPhonemes[j])

    # get only the outputs for valid phrames
    validPredictions = [predictedPhonemes[frame] for frame in valid_frames]

    # return class outputs
    if outputType!= "phonemes":
        predictedPhonemes = [phoneme_set_39[phoneme] for phoneme in predictedPhonemes]
        reducedPhonemes = [phoneme_set_39[phoneme] for phoneme in reducedPhonemes]
        validPredictions = [phoneme_set_39[phoneme] for phoneme in validPredictions]

    return predictedPhonemes, reducedPhonemes, validPredictions


