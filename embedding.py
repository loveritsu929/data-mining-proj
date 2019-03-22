# -*- coding: utf-8 -*-
import pickle, csv
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

glove_size = 200
glove_file = './glove/glove.twitter.27B.200d.txt'
word_vocabulary = [None]
#word_embedding = [[0.0] * glove_size]
word_count = {}
word_vector = {}

spacy_nlp = spacy.load(name="en_core_web_lg", disable=["tagger", "parser", "ner"])

def load_file(file_path, file_type):
    if file_type == "csv":
        with open(file=file_path, mode="rt", encoding="utf-8") as fp:
            lines = list(csv.reader(fp))[1:]
            return lines
        
    elif file_type == "txt":
        with open(file=file_path, mode="rt", encoding="utf-8") as file_stream:
            return file_stream.read().splitlines()
        
    elif file_type == "obj":
        with open(file=file_path, mode="rb") as fp:
            return pickle.load(fp)

    else:
        pass

def dump_data(data_buffer, file_path, file_type):
    if file_type == "txt":
        with open(file=file_path, mode="wt", encoding="utf-8") as fp:
            fp.write("\n".join(data_buffer))
            
    elif file_type == "csv":
        with open(file=file_path, mode='w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(data_buffer)
            
    elif file_type == "obj":
        with open(file=file_path, mode="wb") as fp:
            pickle.dump(obj=data_buffer, file=fp)

    else:
        pass

def sent2vec(word_vector, vocab, tfidf, sent_idx, sent):
    #result = [[0.0] * glove_size]
    result = np.zeros(glove_size)
    sent_len = 0
    num_nan = 0
    for token in spacy_nlp(sent):
        if token.text in word_vector:
            sent_len += 1
            token_idx = vocab.index(token.text)
            result += np.array(word_vector[token.text]) * tfidf[sent_idx][token_idx]
        if sent_len == 0:
            num_nan += 1
            sent_len = 1
    result = result / sent_len
    return result

def convert_emb_to_arff(emb, emb_size, arff_file, data_type):
    header = []
    if data_type in ['train', 'dev', 'test']:
        relation = ['@relation ' + data_type]
        attrs = []
        for i in range(emb_size):
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

#word count
for sent in train_sents:
    for token in spacy_nlp(sent):
        #word_count[token.text] = word_count[token.text] + 1 if token.text in word_count else 1
        word_count[token.text] = word_count[token.text] + 1 if token.text in word_count else 1
#word embedding           
for item in load_file(glove_file, "txt"):
    glove_elements = item.strip().split(" ")

    if glove_elements[0] in word_count:
        word_vector[glove_elements[0]] = [float(element) for element in glove_elements[1:glove_size + 1]]
#build vocab
for word in sorted(word_count, key=word_count.get, reverse=True):
    if word in word_vector:
        word_vocabulary.append(word)
        #word_embedding.append(word_vector[word])            

#compute TF-IDF
vectorizer = TfidfVectorizer(stop_words = None, tokenizer=None, min_df=1, lowercase=True, vocabulary = word_vocabulary)
tfidf = vectorizer.fit_transform(train_sents) 
tfidf_array = tfidf.toarray()
#print(tfidf)
#print(tfidf.toarray())
vocab = vectorizer.get_feature_names() # vocab

# get train emb
train_sents_emb = []
for i,sent in enumerate(train_sents):
    train_sents_emb.append(sent2vec(word_vector, vocab, tfidf_array, i, sent))
train_sents_emb = [sent_vec.tolist() + [train_labels[i]] + [train_ids[i]] for i, sent_vec in enumerate(train_sents_emb)]

# get dev emb
dev_sents_emb = []
for i,sent in enumerate(dev_sents):
    dev_sents_emb.append(sent2vec(word_vector, vocab, tfidf_array, i, sent))
dev_sents_emb = [sent_vec.tolist() + [dev_labels[i]] + [dev_ids[i]] for i, sent_vec in enumerate(dev_sents_emb)]

# get test emb
test_sents_emb = []
for i,sent in enumerate(test_sents):
    test_sents_emb.append(sent2vec(word_vector, vocab, tfidf_array, i, sent))
test_sents_emb = [sent_vec.tolist() + [test_labels[i]] + [test_ids[i]] for i, sent_vec in enumerate(test_sents_emb)]

#dump_data(sents_emb,'emb_demo.csv','csv')
convert_emb_to_arff(train_sents_emb, glove_size, 'train_emb_glove{:d}.arff'.format(glove_size), 'train')
convert_emb_to_arff(dev_sents_emb, glove_size, 'dev_emb_glove{:d}.arff'.format(glove_size), 'dev')
convert_emb_to_arff(test_sents_emb, glove_size, 'test_emb_glove{:d}.arff'.format(glove_size), 'test')
