#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:45:04 2021

@author: binn

"""

#%%
#Read fasta files
from Bio import SeqIO
import xml.etree.ElementTree as ET
import re
from sklearn.model_selection import train_test_split
from sklearn import svm
import re

#%%
#Records
sequences = [sequence.seq for sequence in list(SeqIO.parse("motifs/all.fasta", "fasta"))]

#%%
#Motifs
motifs = []
for idx in range(3, 21):
    for child in ET.parse("motifs/{}_{}/meme.xml".format(idx, idx)).getroot()[2]:
        motifs.append(child[2].text.replace("\n", "").replace("X", "[A-Z]"))
        
#%%
print(motifs[350])
print(re.findall(motifs[350], "KRNQFPPLPMIKDLLWEFYH"))
#%%
count = 0
for motif in motifs[140:160]:
    if "X" in motif:
        count += 1

print (count)

#%%
X1 = []
for sequence in sequences:
    sequence_vector = []
    sequence_vector.append(len(re.findall(motifs[350], str(sequence))))
    X1.append(sequence_vector)
    
print(X1)

#%%
X = []
for sequence in sequences:
    sequence_vector = []
    for motif in motifs:
        sequence_vector.append(len(re.findall(motif, str(sequence))))
    X.append(sequence_vector)
    
#%%
print(X[0][0])
    
#%%
count = 0
for vector in X:
    if (len(vector) == vector.count(0)):
        count += 1

print("Vector 0: {}".format(count))

#%%
count_m = 0
for vector in X:
    count_m += vector[350]
    
print(count_m)

#%%
y = [0] * len(bombali_records) + [1] * len(bundi_records) + [2] * len(tai_records)
#print(y)

#%%
X_train, X_test, y_train, y_test = train_test_split(result_vector, y, train_size=0.8, random_state = 0)

#%%

clf = svm.SVC()

clf.fit(X_train, y_train)

clf.predict(X_test)

print(clf.score(X_test, y_test))

#%%
#print(clf.predict([result_vector[0]]))















