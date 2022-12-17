#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import concatenate

from keras.callbacks import EarlyStopping

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys


# In[2]:


# Function to extract sequence features -- BLOSUM62
def BLOSUM62(sequences):
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2, -3, 1,  0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6, -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3, -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '*': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # *
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # X
        'U': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # U
        '_': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # _
    }
    encodings = []
    for sequence in sequences:
        code=[]  
        for j in sequence:
            code = code + blosum62[j]
        encodings.append(np.array(code))       
    return encodings

# Function to assign pathway and PPI encoding to corresponding items
def Embedding(feature, data, D):

    item =[]
    for i in range(0, len(data)):
        if data["SubID"][i] in feature.keys():
            item.append(feature[data["SubID"][i]])
        else:
            item.append(np.array([0]*D))
            
    return item


# In[3]:


# python phosphorylation_prediction.py SVM Q9Y243 input.csv
model_type = sys.argv[1]
kinase = sys.argv[2]
filename = sys.argv[3]


# In[4]:


try:
    df = pd.read_csv(filename)
except:
    print("File name error: the name of input file should be specified as input.csv !")


# In[5]:


if model_type == "SVM":
    X_seq_validate = np.array(BLOSUM62(df["sequence"]))
    filename="../3_pretrained models/SVM/"+ kinase +"_SVM_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(X_seq_validate)
    df["predict (1-yes and 0-no)"] = result
    df.to_csv("./predicted results by kinase"+ kinase + " SVM.csv")
    
    
elif model_type == "DL":
    
    # a dictionary
    pathway = np.load("../1_features/path_embedding.npy", allow_pickle=True)
    pathway = pathway.flat[0]

    # a dictionary
    PPI = np.load("../1_features/sdne_embedding.npy", allow_pickle=True)
    PPI = PPI.flat[0]    
    
    #Sequence
    X_seq_validate = np.array(BLOSUM62(df["sequence"]))
    #ppi
    X_ppi_validate = np.array(Embedding(PPI, df, 128))
    #pathway
    X_path_validate = np.array(Embedding(pathway, df, 347))
    
    filename="../3_pretrained models/FCNN_LSTM/"+ kinase +"_DL_model.sav"
    model_combined = pickle.load(open(filename, 'rb'))    

    validate_output = model_combined.predict([X_seq_validate.reshape(len(X_seq_validate), 1, 300), X_path_validate, X_ppi_validate])
    pred = np.array(np.concatenate(np.where(validate_output > 0.5, 1, 0)).flat)   
    df["predict (1-yes and 0-no)"] = pred
    df.to_csv("./predicted results by kinase"+ kinase + " FCNN_LSTM model.csv")  
    
    
else:
    print("Error: Model type should be specified as either SVM or DL!")


# In[ ]:




