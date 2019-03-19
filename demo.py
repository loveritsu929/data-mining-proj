# -*- coding: utf-8 -*-

import csv 
file = './test.csv'

with open(file,'r') as fp:
    lines = list(csv.reader(fp))
    #print(type(lines))
    for line in lines[0:2]:
        # 3-element list, text + label(string) + id
        print(line)