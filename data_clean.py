# -*- coding: utf-8 -*-
import re
import csv
#a = '"1/25/2013Nerd  --*+   ....         +--****tech??!!""'
def clean(string):
    
    string = re.sub(r'[\*\"]', ' ', string)
    string = re.sub(r'[\#]', ' ', string)
    string = re.sub(r'[+-]', ' ', string)
    string = re.sub(r'[\.]{2,}', '.', string)
    string = re.sub(r'[?]{2,}', '?', string)
    string = re.sub(r'[!]{2,}', '!', string)
    
    string = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', string) # date
    string = re.sub(r'\s{2,}' ,' ',string)
    string = string.strip()
    return string

#b = clean(a)
#print(b)

#text,class,ID for train&dev; ID, text for test
for file in ['myTrain', 'myDev']:
    output_file = file + '_cleaned.csv'
    data = []
    with open(file + '.csv', 'r') as fp:
        lines = list(csv.reader(fp))
        for line in lines[1:]:
            data.append([clean(line[0]), line[1], line[2]])
            
    with open(output_file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['text', 'class', 'ID'])
        writer.writerows(data)
        
#ID, text for test       
for file in ['test']:
    output_file = file + '_cleaned.csv'
    data = []
    with open(file + '.csv', 'r') as fp:
        lines = list(csv.reader(fp))
        for line in lines[1:]:
            data.append([line[0], clean(line[1])])
            
    with open(output_file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['ID', 'text'])
        writer.writerows(data)