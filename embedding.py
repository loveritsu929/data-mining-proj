# -*- coding: utf-8 -*-
import pickle, csv
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

glove_size = 50
glove_file = ' ' #TODO
word_vocabulary = [None]
#word_embedding = [[0.0] * glove_size]
word_count = {}
word_vector = {}

spacy_nlp = spacy.load(name="en_core_web_lg", disable=["tagger", "parser", "ner"])

def load_file(file_path, file_type):
    if file_type == "csv":
        with open(file=file_path, mode="rt", encoding="utf-8") as fp:
            lines = list(csv.reader(fp))[1:]
            return lines[1:] 
        
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
        with open(file=file_path, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(data_buffer)
            
    elif file_type == "obj":
        with open(file=file_path, mode="wb") as fp:
            pickle.dump(obj=data_buffer, file=fp)

    else:
        pass

def sent2vec(word_vector, vocab, tfidf, sent_idx, sent):
    result = [[0.0] * glove_size]
    sent_len = 0
    for token in spacy_nlp(sent):
        if token in word_vector:
            sent_len += 1
            token_idx = vocab.index(token)
            result += word_vector[token] * tfidf[sent_idx][token_idx]
    result = result / sent_len
    return result
  
#if __name__ == '__main__':
temp = load_file('myTrain_cleaned.csv', 'csv')[:3] + load_file('myDev_cleaned.csv', 'csv')[:3]
sents = [item[0] for item in temp]
for sent in sents:
    for token in spacy_nlp(sent):
        #word_count[token.text] = word_count[token.text] + 1 if token.text in word_count else 1
        word_count[token.lemma_] = word_count[token.lemma_] + 1 if token.text in word_count else 1
            
for item in load_file(glove_file, "txt"):
    glove_elements = item.strip().split(" ")

    if glove_elements[0] in word_count:
        word_vector[glove_elements[0]] = [float(element) for element in glove_elements[1:glove_size + 1]]

for word in sorted(word_count, key=word_count.get, reverse=True):
    if word in word_vector:
        word_vocabulary.append(word)
        #word_embedding.append(word_vector[word])            

vectorizer = TfidfVectorizer(stop_words = None, tokenizer=spacy_nlp, lowerCase=True)
tfidf = vectorizer.fit_transform(sents) 
print(tfidf)
print(tfidf.toarray())
vocab = vectorizer.get_feature_names() # vocab