# MIT License
#
# Copyright (c) 2016 matthijs van keirsbilck
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import print_function
import re

# read the phonemes in a list, strip newlines
vocab = [line.rstrip('\n') for line in open('./background/phonemeList.txt')]
print("Possible phonemes are: ",vocab)
# create dict: keys= phonemes, values= nb of occurences
countLipspeakers = dict((x,0) for x in vocab)

## read MLFfiles, update the counters (values) of the dict
# lipspeaker file
lines = [line.rstrip('\n') for line in open('./MLFfiles/lipspeaker_labelfiles.mlf')]
for line in lines:

    # # only count lipspkr1
    # if "Lipspkr2" in str(line):
    #     break

    for w in re.findall(r"\w+", line):  # loop through the words of the line
        if w in countLipspeakers:       # if a phoneme, add 1 to the value corresponding to the phoneme (key)
            countLipspeakers[w] += 1


# read the volunteer file
countVolunteers = dict((x,0) for x in vocab)

lines = [line.rstrip('\n') for line in open('./MLFfiles/volunteer_labelfiles.mlf')]
for line in lines:
    for w in re.findall(r"\w+", line):
        if w in countVolunteers:
            countVolunteers[w] += 1

# adds the values of the dicts together
from collections import Counter
a = countLipspeakers
b = countVolunteers
# add dict values. Normal a.items + b.items overwrites instead of adding
# second answer: http://stackoverflow.com/questions/11011756/is-there-any-pythonic-way-to-combine-two-dicts-adding-values-for-keys-that-appe#11011846
total = dict(a.items() + b.items() +
    [(k, a[k] + b[k]) for k in set(b) & set(a)])

# Do some nice printing
print("###########################")
print("For the lipspeakers: ", countLipspeakers)
print("------------------")
print ('For the volunteers: ', countVolunteers)
print("------------------")
print ('For the whole database: ', total)
print("------------------")


# list that contains the key-value tuple pairs, sorted by value
totalSorted = []
for key, value in sorted(total.iteritems(), key=lambda (k,v): (v,k)):
    totalSorted.append((key, value))

# print ten first and ten last values
print('Ten least often occuring: ', totalSorted[0:10])
print('Ten most often occuring : ', totalSorted[-10:])

# print total number of phonemes found
lipspeakerSum = sum(countLipspeakers.values())
volunteerSum = sum(countVolunteers.values())
print("------------------")
print("nb of phonemes for the lipspeakers: \t", lipspeakerSum)
print("nb of phonemes for the volunteers: \t", volunteerSum)

totalSum = sum(total.values())
print('Total number of phonemes in database: \t', totalSum)
print('Avg number of images per phonemes: \t', totalSum/39)
print('Median number of images per phoneme: ', totalSorted[20])
