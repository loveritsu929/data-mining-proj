# -*- coding: utf-8 -*-

import csv 
from sklearn.model_selection import train_test_split

file = './train.csv'
numPos = 0
numNeg = 0
numNeu = 0

with open(file,'r') as fp:
    lines = list(csv.reader(fp))
    #print(type(lines))
    numSamples = len(lines) - 1
    for line in lines[1:]:
        # 3-element list, text + label(string) + id
        #print(line)
        assert line[1] == 'positive' or line[1] == 'negative' or line[1] == 'neutral'
        if line[1] == 'positive':
            numPos += 1
        elif line[1] == 'negative':
            numNeg += 1
        else:
            numNeu += 1

print('Total num = {}, numPos = {}, numNeg ={}, numNeu = {}'.format(numSamples, numPos, numNeg, numNeu))
print('Original Ratios: pos = {:.4f}, neg = {:.4f}, neu = {:.4f}'.format(numPos/numSamples, numNeg/numSamples, numNeu/numSamples))

numDev = 5000
# pos : neg : neu = 2793 : 1350 : 856
print('If #dev = {}, then devPos = {}, devNeg = {}, devNeu = {}'.format(numDev, int(numPos/numSamples*numDev), int(numNeg/numSamples*numDev), int(numNeu/numSamples*numDev)))

outputTrain = './myTrain.csv'
outputDev = './myDev.csv'
X_train, X_test, Y_train, Y_test = train_test_split(lines[1:], [line[1] for line in lines[1:]], test_size=5000, random_state=42)

samplePos = sampleNeg = sampleNeu = 0
for sample in X_test:
    assert sample[1] == 'positive' or sample[1] == 'negative' or sample[1] == 'neutral'
    if sample[1] == 'positive':
        samplePos += 1
    elif sample[1] == 'negative':
        sampleNeg += 1
    else:
        sampleNeu += 1
print('Dev Ratios: pos = {:.4f}, neg = {:.4f}, neu = {:.4f}'.format(samplePos/5000, sampleNeg/5000, sampleNeu/5000))       

with open(outputTrain, 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerows(X_train)
    
with open(outputDev, 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerows(X_test)