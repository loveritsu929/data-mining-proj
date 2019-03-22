# -*- coding: utf-8 -*-
from bert_serving.client import BertClient
import numpy as np
from embedding import load_file
import csv

bert = BertClient(check_length=False)

def sent2vec(sent):
    emb= bert.encode([sent])
    return emb

def convert_emb_to_arff(emb, arff_file, data_type):
    header = []
    if data_type in ['train', 'dev', 'test']:
        relation = ['@relation ' + data_type]
        attrs = []
        for i in range(1024):
            attr = ['@attribute a{:d} numeric'.format(i+1)]
            attrs.append(attr)
        attrs.append(['@attribute class {positive, neutral, negative}'])
        attrs.append(['@attribute id numeric'])
        attrs.append(['@data'])

            
    header.append(relation)
    header += attrs
    with open(arff_file, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(relation)
            writer.writerows(attrs)
            writer.writerows(emb)
            
#if __name__ == '__main__':
train = load_file('myTrain_cleaned.csv', 'csv')
dev = load_file('myDev_cleaned.csv', 'csv')
test = load_file('test_cleaned.csv', 'csv')
# train data
train_sents = [item[0] for item in train]
train_labels = [item[1] for item in train]
train_ids = [item[2] for item in train]
# dev data
dev_sents = [item[0] for item in dev]
dev_labels = [item[1] for item in dev]
dev_ids = [item[2] for item in dev]
#test data
test_sents = [item[1] for item in test]
test_ids = [item[0] for item in test]
test_labels = ['neutral'] * 10000# fake labels

# get train emb
train_sents_emb = []
for i,sent in enumerate(train_sents):
    train_sents_emb.append(sent2vec(sent))
train_sents_emb = [sent_vec.tolist() + [train_labels[i]] + [train_ids[i]] for i, sent_vec in enumerate(train_sents_emb)]
train_sents_emb = [item[0] +[item[1]] +[item[2]] for item in train_sents_emb]

# get dev emb
dev_sents_emb = []
for i,sent in enumerate(dev_sents):
    dev_sents_emb.append(sent2vec(sent))
dev_sents_emb = [sent_vec.tolist() + [dev_labels[i]] + [dev_ids[i]] for i, sent_vec in enumerate(dev_sents_emb)]
dev_sents_emb = [item[0] +[item[1]] +[item[2]] for item in dev_sents_emb]

# get test emb
test_sents_emb = []
for i,sent in enumerate(test_sents):
    test_sents_emb.append(sent2vec(sent))
test_sents_emb = [sent_vec.tolist() + [test_labels[i]] + [test_ids[i]] for i, sent_vec in enumerate(test_sents_emb)]
test_sents_emb = [item[0] +[item[1]] +[item[2]] for item in test_sents_emb]

#dump_data(sents_emb,'emb_demo.csv','csv')
convert_emb_to_arff(train_sents_emb, 'train_emb_bert{:d}.arff'.format(1024), 'train')
convert_emb_to_arff(dev_sents_emb, 'dev_emb_bert{:d}.arff'.format(1024), 'dev')
convert_emb_to_arff(test_sents_emb, 'test_emb_bert{:d}.arff'.format(1024), 'test')