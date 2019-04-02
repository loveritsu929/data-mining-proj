#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:08:03 2019

@author: cxing95
"""
import csv
import numpy as np

ids = []
label_map = {2:'positive', 1:'neutral', 0:'negative'}

with open('test.csv', 'r') as fp:
    lines = list(csv.reader(fp))
    for line in lines[1:]:
        ids.append(line[0])
        
output = np.load('output_ids.npy')
assert len(output) == len(ids)
with open('prediction.txt', 'w') as fp:
    fp.write('REVIEW-ID CLASS\n')
    for i, cid in enumerate(ids):
        fp.write(cid + ' ' + label_map[output[i]] + '\n')