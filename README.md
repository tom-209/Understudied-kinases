# Holistic similarity-based prediction of phosphorylation sites for understudied kinases

### Description

This repository contains the data and Jupyter notebook files used in the paper mentioned in the `Citation request` 
section below. 
The selection criteria leveraged to identify highly similar kinases with respect to understudied kinases 
are included in the supplementary table of the paper.
The Machine Learning-driven model `SVM` was used as a first pass and trained on sequence-related data only, and 
the Deep Learning-driven model (FCNN-LSTM) was leveraged as a second pass.

### Repository structure and usage

```
.
├── 0_data                                        # Folder containing five data-related files
│   ├── fused kinase_kinase similarity.csv        # The fuse Kinase-Kinase similarities
│   ├── Highly_Similar_Kinases.pickle             # Highly similar kinases to the 82 understudied kinases
│   ├── Human K_S data.csv                        # Experimentally-verified kinase-specific kinase-substrate pairs
│   ├── negative ST sites.csv                     # Non-phosphorylation ST sites regarded as negative sites
│   ├── negative Y sites.csv                      # Non-phosphorylation Y sites regarded as negative sites
│   └── understudied_kinases.csv                  # List of the 116 understudied kinases and their categories
│
├── 1_features                                    # The encoded KEGG pathway and PPI (encoded with SDNE) features for human proteins
│
├── 2_model training                              # Folder containing Jupyter notebook file and .py file
│   ├── 0_data_processing.ipynb                   # File used to identify the list of highly similar kinases with respect to understudied kinases
│   └── 1_model training.py                    # File on the nested SVM and FCNN_LSTM predictive model
│
├── 3_pretrained models                           # Folder containing pretrained models
│   ├── SVM                                       # pretrained SVM models
│   └── FCNN_LSTM                                 # pretrained DL models
├── 4_phosphorylation prediction                 # Folder containing model testing files
│   ├── input.csv                                 # user input file
│   └── phosphorylation_prediction.py             # model prediction file
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```

### Quick Start: Setup Your Environment
To begin, install the environment using:

"conda env create -f environment.yml"

This will take a bit of time to run.

Please note that for the code above to work, you need to be in the directory where the environment.yml file stays

Activate the environment that you'd like to use
Conda 4.6 and later versions (all operating systems):

"conda activate phosphorylation-python"

The environment name is phosphorylation-python as defined in the environment.yml file.


### Run training models
Users can go to directory "2_model training" and run:
"python 1_model training.py"
A result table will generate in the working directory.


### Run pretrained model on your own data
To test if specific phosphorylation sites can be phosphorylation by certain kinase, users should preprate the data with the same format of 
"4_phosphorylation prediction/input.csv", namely, the 15-mer sequence, and its protein uniprot ID.

To run the test, users also specify the model type (either "SVM" or "DL") and the kinase type (should be one KIN_ACC of the 82 understudied kinases, as
listed in the Table S1 in the paper). With the model type specified, the program will call the pretrained SVM or FCNN_LSTM models stored in directory
"3_pretrained models"

For example:
"python phosphorylation_prediction.py SVM Q9Y243 input.csv"
or
""python phosphorylation_prediction.py DL Q9Y243 input.csv""

After running, a result file will generate and be store in working directory (with 1 indicating a predicted phosphorylation sites and 0 suggesting not a
predicted phosphorylation site).


### Citation request

If you use these data and/or codes, please cite the following paper currently under review:

- "Renfei Ma, Shangfu Li, Luca Parisi, Wenshuo Li, Hsien-Da Huang, and Tzong-Yi Lee. Holistic similarity-based prediction of phosphorylation
sites for understudied kinases." 
