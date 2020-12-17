# Introduction
This repository aims to show the result and the methodology used to achieve the objectives of the technical test. The task is based on natural language processing, using Python programming language for their development.

# Getting Started

1. Installation process:
    * Install conda
    * conda create -n testseedtag python=3.6
    * conda activate testseedtag
    * pip install -r requirements.txt
    * python -m nltk.downloader all
    * python -m spacy download en_core_web_sm
    
1. Software dependencies
    * All python packages needed are included in `requirements.txt`

1. Train the model:
    * If step 1 is OK you can run the line's code below to train the model.
    * Go to **scripts** folder 
    ```
    cd code\scripts
    ```
    * Run the following line in cmd
    
    ```
    python train.py ..\..\dataset\
    ```
    * The model was saving in the folder models as **model.h5**.
1. Test the model
    * If step 3 is OK you can run the line's code below to test the model.
    * Go to **scripts** folder 
    ```
    cd code\scripts
    ```
    * Run the following line in cmd
    
    ```
    python classify.py model.h5 54564 60841
    ```


# Methodology
The repository is structured in the following folders:
* code: You will find the forlder scripts that contains the python scripts and a folder called "notebooks".

* dataset: You will find the folders categorized by documents.

**Generate Dataset**

1. A dataset was created with all the files that are in the dataset folder
    * To view the Dataset go to **Analysis.ipynb** 

**OBJECTIVE 1**: Create a document categorization classifier for the different contexts of the documents

**Pre-process**

To perform pre-processing, a class called "preprocess_data.py" was developed. This class has different methods:

* clean_dataframe: It was used to clean / pre-process the data, where null values, duplicate values, stopwords ​​and any other noise such as digits and punctuation marks are eliminated. Also, words with less than three characters were removed.
* To view the other methods go to notebooks: **Analysis.ipynb**

**Model**

 Classifier:

For this task, the **Embedding**  was used to transform text into a features to train the CNN model. This method consists in represent a word in a vector in this opportunity Glove6B of dimension 100 was used. Moreover the **TFIDF** (Term frequency – Inverse document frequency) was used to transform text into a features to train the LSTM  model. This method consists of creating a matrix that contains the frequency of the words found in each document.

 * Two neural networks were trained. 
One **CNN** (Convolutional Neural Network) and the other **LSTM** (Long short term memory). In the notebook **Analysis.ipynb** the performance of both models is appreciated, the comparison is made using the precision vs recall curve metric. 

In this case, by comparing the two models, the CNN achieved better results for all classes using the metric mentioned before, for this reason it was chosen to make the final model and be tested in script classify.

# CONCLUSION

In accordance with the objectives set out in the test, two types of neural networks were used to classify the text. The results showed that the optimal way to classify this type of text is to use CNN and Embedding because CNN is very well suited to dimension problems with large matrix.

It was decided to use the precision vs recall curve metric to measure the performance of the model optimally in each class. 

