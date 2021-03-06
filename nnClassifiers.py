#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:43:40 2019

@author: cxing95
"""

import csv, torch, sys, copy, os
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#hyperparameters
lnR = 1e-4
numEpoch = 2000

file_abbr = sys.argv[1]
starting_idx = {'glove50': 54,'glove100': 104, 'glove200': 204, 'bert1024': 1028}
data_dims = {'glove50': 50,'glove100': 100, 'glove200': 200, 'bert1024': 1024}
labels = ['negative','neutral','positive']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def convert_label(label):
    idx = labels.index(label)
    assert idx in [0,1,2]
    return idx

def get_file(abbr, mode):
    train_file = 'train_emb_' + abbr + '.arff'
    dev_file = 'dev_emb_' + abbr + '.arff'
    test_file = 'test_emb_' + abbr + '.arff'
    
    if mode == 'train': 
        file_name = train_file
    elif mode == 'dev':
        file_name = dev_file
    else:
        file_name = test_file
        
    return file_name

def load_file(file_name):
    idx = starting_idx[file_abbr]
    with open(file_name, 'r') as fp:
        lines = list(csv.reader(fp))[idx:]       
    return lines
    
class EmbDataset(data.Dataset):
    ''' a dataset for my sentence embedding
    '''
    
    def __init__(self, mode, file_abbr):
        self.mode = mode
        self.file_abbr = file_abbr
        self.file_name = get_file(file_abbr, mode)
        self.fp = load_file(self.file_name)
        self.dataset = self.make_dataset()
        
    def make_dataset(self):
        ds = []
        for line in self.fp:
            id = line[-1]
            label = convert_label(line[-2])
            data = line[:-2]
            data = [float(i) for i in data]
            data = torch.as_tensor(data)
            ds.append((data,label))
            
        return ds
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
class MyNN(nn.Module):
    
    def __init__(self, abbr):
        super(MyNN, self).__init__()
        self.dim = data_dims[abbr]
        if self.dim == 50:
            self.fc1 = nn.Linear(50, 20)
            self.fc2 = nn.Linear(20, 3)
        elif self.dim == 100:
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 3)
        elif self.dim == 200:
            self.fc1 = nn.Linear(200, 100)
            self.fc2 = nn.Linear(100, 3)
        elif self.dim == 1024:
            self.fc1 = nn.Linear(1024, 256)
            self.fc2 = nn.Linear(256, 3)
        else:
            print('invalid dimension!')
        
        self.dropout = nn.Dropout(p = 0.5)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        
        h = self.fc1(x)
        h = self.relu(h)
        
        h = self.dropout(h)
        
        h = self.fc2(h)
        h = self.relu(h)
        
        return h

datasets = {}
train_ds = EmbDataset('train', file_abbr)
dev_ds = EmbDataset('dev', file_abbr)
datasets['train'] = train_ds
datasets['dev'] = dev_ds
dataloaders = {}
dataloaders['train'] = data.DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
dataloaders['dev'] = data.DataLoader(dev_ds, batch_size=1024, shuffle=False, num_workers=0)

model = MyNN(file_abbr)
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lnR, amsgrad = True)

#train and validata
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
loss_dict = {}
loss_dict['train'] = []
loss_dict['dev'] = []
acc_dict = {}
acc_dict['train'] = []
acc_dict['dev'] = []


print('Start training, using dataset: {}'.format(file_abbr))
print('NN structure: ')
print(model)
for epoch in range(numEpoch):
    print('Epoch {}/{}'.format(epoch+1, numEpoch))
    print('-' * 20)
    
    for phase in ['train','dev']:
        if phase  == 'train':
            model.train()
        else: 
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0
        
        for i, (sample, label) in enumerate(dataloaders[phase]):
            #move to gpu
            sample = sample.to(device)
            label = label.to(device)
            
            #clear gradients
            optimizer.zero_grad()
            
            #forward
            with torch.set_grad_enabled(phase == 'train'):
                output = model(sample)
                _, preds = torch.max(output, 1)
                loss = loss_func(output, label)
                
                #backprop
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
            running_loss += loss.item() * sample.size(0)
            running_corrects += torch.sum(preds == label.data)           
            print('Epoch {} Iteration {}: running_corrects: {} running loss = {:4f}'.format(epoch+1,i,running_corrects,running_loss))
            
        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects.double() / len(datasets[phase])
        loss_dict[phase].append(epoch_loss)
        acc_dict[phase].append(epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print(' ')
        
        #save the best model
        if phase == 'dev' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join('./my-models', 'bestmodel_{}.mdl'.format(file_abbr)))
print('Finish Training, Best val Acc: {:4f}'.format(best_acc))

#plot
x = list(range(1, numEpoch+1))
plt.figure(file_abbr)
plt.subplot(2, 2, 1).set_title('train loss')
plt.plot(x, loss_dict['train'])
plt.subplot(2, 2, 2).set_title('train acc')
plt.plot(x, acc_dict['train'], color = 'red',marker='+', linestyle='dashed')
plt.subplot(2, 2, 3).set_title('dev loss')
plt.plot(x, loss_dict['dev'])
plt.subplot(2, 2, 4).set_title('dev acc')
plt.plot(x, acc_dict['dev'], color = 'red',marker='+', linestyle='dashed')
plt.show()
