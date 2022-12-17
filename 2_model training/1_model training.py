#!/usr/bin/env python
# coding: utf-8

# # 0. Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import pickle


# In[ ]:


from sklearn.metrics import roc_auc_score
from statistics import mean


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import concatenate

from keras.callbacks import EarlyStopping


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# # 1. Functions

# In[ ]:


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

# Function to select negative data based on the composition of positive data
def Negative(df_neg_ST, df_neg_Y, positive):
    
    # The number of ST/Y sites is consistent in positive and negative data
    if "Y" in positive["sequence"].str[7].value_counts().keys():
        num_Y = positive["sequence"].str[7].value_counts()["Y"]
    else:
        num_Y =0
    num_ST = len(positive)- num_Y

    negative_ST = df_neg_ST.sample(n=num_ST).reset_index(drop=True)
    negative_Y = df_neg_Y.sample(n=num_Y).reset_index(drop=True)
    negative = pd.concat([negative_ST, negative_Y], axis=0).reset_index(drop=True).sample(frac=1).reset_index(drop=True)

    negative["label"]=[0]*len(negative)

    #combine positive and negative data
    negative = negative.rename(columns={'protein neg': 'SubID', 'sequence neg': 'sequence'}) 
    
    return negative


# Function to assign pathway and PPI encoding to corresponding items
def Embedding(feature, data, D):

    item =[]
    for i in range(0, len(data)):
        if data["SubID"][i] in feature.keys():
            item.append(feature[data["SubID"][i]])
        else:
            item.append(np.array([0]*D))
            
    return item
        
    
def FCNN_LSTM(X_seq_train, X_path_train, X_ppi_train, Y_train, X_seq_validate, X_path_validate, X_ppi_validate):

    #=======================
    # 3.1 Deep Learning-driven model
    input_pathway = Input(shape=(347,))
    input_ppi = Input(shape=(128,))

    # Pathways
    model_pathway = Sequential()
    model_pathway = Dense(32, activation="relu")(input_pathway)
    model_pathway = Dense(4, activation="relu")(model_pathway)
    model_pathway = Model(inputs=input_pathway, outputs=model_pathway)

    # PPI
    model_ppi = Sequential()
    model_ppi = Dense(32, activation="relu")(input_ppi)
    model_ppi = Dense(4, activation="relu")(model_ppi)
    model_ppi = Model(inputs=input_ppi, outputs=model_ppi)

    # Sequences
    model_seq = Sequential()
    model_seq.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(1, 300)))
    model_seq.add(LSTM(100, activation='relu', return_sequences=True))
    model_seq.add(LSTM(50, activation='relu', return_sequences=True))
    model_seq.add(LSTM(25, activation='relu'))
#        model_seq.add(Dense(20, activation='relu'))
#        model_seq.add(Dense(10, activation='relu'))


    combined = concatenate([model_seq.output, model_pathway.output, model_ppi.output])
    # Apply a fully-connected (FC) layer and then a regression prediction on the
    # combined outputs
    z = Dense(2, activation="relu")(combined)
    z = Dense(1, activation="sigmoid")(z)

    # Then, output a single value
    model_combined = Model(inputs=[model_seq.input, model_pathway.input, model_ppi.input], outputs=z)
    model_combined.compile(optimizer='adam', loss='mse')

    # Early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    model_combined.fit([X_seq_train.reshape(len(X_seq_train), 1, 300), X_path_train, X_ppi_train], Y_train, epochs=300, validation_split=0.2, batch_size=128, verbose=0, callbacks=[es])   


    validate_output = model_combined.predict([X_seq_validate.reshape(len(X_seq_validate), 1, 300), X_path_validate, X_ppi_validate])

    return(np.array(np.concatenate(np.where(validate_output > 0.5, 1, 0)).flat))   

    

def SVM(X_seq_train, Y_train, X_seq_validate):
    # 3.3 SVM model
    SVM_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    SVM_clf.fit(X_seq_train, Y_train)
#     preds = SVM_clf.predict(X_seq_test)
    return(SVM_clf.predict(X_seq_validate))


    


# # 2. Read data

# ### 2.1 Read understudied kinases and their highly similar kinases

# In[ ]:


with open('../0_data/Highly_Similar_Kinases.pickle', 'rb') as handle:
    dict_N = pickle.load(handle)


# In[ ]:





# ### 2.2 Read human K-S data and negative data

# In[ ]:


df = pd.read_csv("../0_data/Human K_S data.csv")


# In[ ]:


df_neg_Y = pd.read_csv("../0_data/negative Y sites.csv")
df_neg_ST = pd.read_csv("../0_data/negative ST sites.csv")


# ### 2.3 Read pre-calculated pathway and ppi embeddings 

# In[ ]:


# a dictionary
pathway = np.load("../1_features/path_embedding.npy", allow_pickle=True)
pathway = pathway.flat[0]

# a dictionary
PPI = np.load("../1_features/sdne_embedding.npy", allow_pickle=True)
PPI = PPI.flat[0]


# In[ ]:





# # 3. FCNN+LSTM model 

# In[ ]:


KIN = []
K_accuracy2 = []
K_precision2 = []
K_recall2 = []
K_F1score2 = []
K_AUC2 = []

len_pos = []
model_type = []


for k in dict_N.keys():
    
    # Define empty lists to store performance metrics 
    Accuracy2 = []
    Precision2 = []
    Recall2 = []
    F1_score2 = []
    AUC2 = []

    # 3.1.1 Positive training data --- psites of highly similar kinases
    df1 = df[df["KIN_ACC"].isin(dict_N[k])].reset_index(drop=True)
    positive = df1[["SUB_ACC","PEPTIDE"]].reset_index(drop=True)
    positive["label"] = [1]*len(positive)
    positive = positive.rename(columns={'SUB_ACC': 'SubID', 'PEPTIDE': 'sequence'}) 
    
    
    #3.1.2 validate data --- psites of understudied kinases
    df_val = df[df["KIN_ACC"] ==k].reset_index(drop=True)
    val = df_val[["SUB_ACC","PEPTIDE"]].reset_index(drop=True)
    val["label"] = [1]*len(val)
    val = val[val["PEPTIDE"].str.len() == 15].reset_index(drop=True)
    val_positive = val.rename(columns={'SUB_ACC': 'SubID', 'PEPTIDE': 'sequence'})
        
    ###################################################################################################
    # the SVM was leveraged as a first pass and trained on sequence-related data only, and if its
    # predictive performance was not deemed satisfactory, the DL model (FCNN-LSTM) was used
    ###################################################################################################
    
    ##################################
    # 3.2 first passs (SVM)
    # For each understudied kinase, the process is repeated 10 times to get average performance 
    for i in range(0, 10):
        
        # 3.2.1 Negative training data --- different every time
        negative = Negative(df_neg_ST, df_neg_Y, positive)
        
        val_negative = Negative(df_neg_ST, df_neg_Y, val_positive)
        
        
        # 3.2.2 Train and test splits
        df_input = pd.concat([positive, negative], axis=0).reset_index(drop=True)
        df_input = df_input[df_input["sequence"].str.len() == 15].reset_index(drop=True)
        train, test = train_test_split(df_input, test_size=0.2, random_state=42, shuffle=True)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        
        df_input_val = pd.concat([val_positive, val_negative], axis=0).reset_index(drop=True)
        validate = df_input_val.sample(frac=1).reset_index(drop=True)

        # 3.2.3 Data encoding
        #Sequence data
        X_seq_train = np.array(BLOSUM62(train["sequence"]))
        X_seq_test = np.array(BLOSUM62(test["sequence"]))
        X_seq_validate = np.array(BLOSUM62(validate["sequence"]))

        Y_train = np.array(train["label"])
        Y_test = np.array(test["label"])
        Y_validate = np.array(validate["label"])   
        
        #======================================================================================
        # 3.2.4 the SVM was leveraged as a first pass and trained on sequence-related data only
        target_names = ['neg 0', 'pos 1']
        pred = SVM(X_seq_train, Y_train, X_seq_validate)
        report = classification_report(Y_validate, pred, target_names=target_names, output_dict=True)
        #======================================================================================
   
        target_names = ['neg 0', 'pos 1']
        acc2 = round(report["accuracy"],3)
        pre2 = round(report["weighted avg"]["precision"],3)
        recall2 = round(report["weighted avg"]["recall"],3)
        f12 = round(report["weighted avg"]["f1-score"],3)
        auc2 = roc_auc_score(Y_validate, pred, average='weighted')

        Accuracy2.append(acc2)
        Precision2.append(pre2)
        Recall2.append(recall2)
        F1_score2.append(f12)
        AUC2.append(auc2)
        
    m_type="SVM"    
    
    
    ##################################
    # 3.3 first passs (SVM)
    # if predictive performance from SVM was not deemed satisfactory (accuracy < 0.7), the DL model (FCNN-LSTM) 
    # was used, with KEGG pathways and protein-protein interactions added as additional features
    
    if mean(Accuracy2)<0.7:
    
        # For each understudied kinase, the process is repeated 10 times to get average performance 
        for i in range(0, 10):

            # 3.3.1 negative training data --- different every time
            negative = Negative(df_neg_ST, df_neg_Y, positive)

            val_negative = Negative(df_neg_ST, df_neg_Y, val_positive)


            # 3.3.2 train test split
            df_input = pd.concat([positive, negative], axis=0).reset_index(drop=True)
            df_input = df_input[df_input["sequence"].str.len() == 15].reset_index(drop=True)
            train = df_input.sample(frac=1).reset_index(drop=True)


            df_input_val = pd.concat([val_positive, val_negative], axis=0).reset_index(drop=True)
            validate = df_input_val.sample(frac=1).reset_index(drop=True)


            # 3.3.3 Data encoding
            # Sequence data
            X_seq_train = np.array(BLOSUM62(train["sequence"]))
            X_seq_validate = np.array(BLOSUM62(validate["sequence"]))
            #-----------------------------------------------

            # ppi
            X_ppi_train = np.array(Embedding(PPI, train, 128))
            X_ppi_validate = np.array(Embedding(PPI, validate, 128))

            # pathway
            X_path_train = np.array(Embedding(pathway, train, 347))
            X_path_validate = np.array(Embedding(pathway, validate, 347))
            #-----------------------------------------------

            Y_train = np.array(train["label"])
            Y_validate = np.array(validate["label"])
            #------------------------------------------------


            #========================================================================================
            # the DL model (FCNN-LSTM) as a second pass
            target_names = ['neg 0', 'pos 1']
            pred = FCNN_LSTM(X_seq_train, X_path_train, X_ppi_train, Y_train, X_seq_validate, X_path_validate, X_ppi_validate) 
            report = classification_report(Y_validate, pred, target_names=target_names, output_dict=True)
            #========================================================================================

            acc2 = round(report["accuracy"],3)
            pre2 = round(report["weighted avg"]["precision"],3)
            recall2 = round(report["weighted avg"]["recall"],3)
            f12 = round(report["weighted avg"]["f1-score"],3)
            auc2 = roc_auc_score(Y_validate, pred, average='weighted')

            Accuracy2.append(acc2)
            Precision2.append(pre2)
            Recall2.append(recall2)
            F1_score2.append(f12)
            AUC2.append(auc2)
        m_type="DL"
            
    # 3.4 save results
    KIN.append(k)
    K_accuracy2.append(mean(Accuracy2))
    K_precision2.append(mean(Precision2))
    K_recall2.append(mean(Recall2))
    K_F1score2.append(mean(F1_score2))
    K_AUC2.append(mean(AUC2))
    len_pos.append(len(positive))
    model_type.append(m_type)


# # 4. Results

# In[ ]:


gene={}
for i in range(0, len(df)):
    gene[df["KIN_ACC"][i]] = df["KINASE"][i]

G={}
for i in range(0, len(df)):
    G[df["KIN_ACC"][i]] = df["Group"][i]
    
gene_name = []
group_name = []
num_of_test = []
for k in dict_N.keys():
    gene_name.append(gene[k])
    group_name.append(G[k])
    num_of_test.append(len(df[df["KIN_ACC"] == k].drop_duplicates(subset = ["PEPTIDE"])))
    


# In[ ]:


results = pd.DataFrame()

results["kinase"] = KIN
results["gene names"] = gene_name
results["group"] = group_name
results["test size"] = num_of_test

results["Acc deep"] = K_accuracy2
results["Pre deep"] = K_precision2
results["Recall deep"] = K_recall2
results["F1 score deep"] = K_F1score2
results["AUC deep"] = K_AUC2

results["No of positive sites"] = len_pos
results["Model types"] = model_type


# In[ ]:


results.to_csv("training results.csv")


# In[ ]:




