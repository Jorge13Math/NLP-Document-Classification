#!/usr/bin/python


import json
import logging
import os
import sys
import pandas as pd
import numpy as np
import pickle
import h5py
import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from preprocess_data import Preprocces
sys.path.append(os.path.realpath('../'))
sys.path.append(os.path.realpath('../../'))


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)

class Classify():
    def __init__(self,args):
        self.model = args[0]
        self.files = args[1:]
        self.path_models = '../../models/'
        
        logger.info("Initialized Class")
    
    def load_models(self):

        models = dict()
        

        models['tfidf'] = pickle.load(open(self.path_models+'tfidf.pickle', 'rb'))
        models['cnn'] = load_model(self.path_models+self.model)
        models['label'] = pickle.load(open(self.path_models+'label.pickle','rb'))
        models['tokinezer'] = pickle.load(open(self.path_models + 'tokenizer.pickle','rb'))
        return models

    def get_text(self):
        path = '../../dataset/'
        labels = os.listdir(path)
        logger.info(self.files)
        values = {}
        data = []
        docs = []
        for label in labels:
            if os.path.isdir(path+label):
                for doc in self.files:
                    if os.path.exists(path+label+'/'+doc):
                        logger.info('loading files')

                        f = open(path+label+'/'+doc,'r',encoding='ISO-8859-1')
                        text = f.read()
                        values['Text']=text
                        f.close()
                        values['label']= label
                        data.append(values)
                        values ={}
                        docs.append(doc)
                    
        if not data :
            logger.info(f"The file {doc} not found")
        logger.info(f"files found:{docs}")   
        df = pd.DataFrame(data)
        return df,docs

    def predict(self,df):
        result = {}
        MAX_SEQUENCE_LENGTH=250
        models = self.load_models()
        token = models['tokinezer']
        sequences = token.texts_to_sequences(df['Clean_text'])
        test_pred  = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        model_cnn = models['cnn']
        predict = model_cnn.predict(test_pred)
        for index in df['docs'].index:
            result[df['docs'][index]] = models['label'].inverse_transform(predict)[index]

        return result
        

if __name__ == "__main__":
    
    logger.info(sys.argv[1:])
    classify = Classify(sys.argv[1:])
    

    df , docs= classify.get_text()
    preprocces = Preprocces(df)
    if len(df)>=1:

        logger.info('Cleaning Text')
        df['Clean_text'] = df['Text'].apply(preprocces.clean_text)
        df['docs'] = docs
        
        result = classify.predict(df)
        logger.info('Making predictions:')
        for key,value in result.items():
            print(key,value)
    else:
        pass
    
    
