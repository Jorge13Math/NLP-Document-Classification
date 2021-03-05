#!/usr/bin/python

import os
import sys
import numpy as np
import logging
from create_dataset import genererate_dataframe
from preprocess_data import Preprocces
import pickle
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras import layers
from keras import Input
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.realpath('../'))
sys.path.append(os.path.realpath('../../'))


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100


class Train:
    def __init__(self, path):
        self.data_path = path
        self.path_models = "../../models/"
        logger.info("Initialized Class")
    
    def train_model(self):
        path = self.data_path
        df = genererate_dataframe(path)
        preprocces = Preprocces(df)
        df = preprocces.clean_dataframe()
        stats_words = preprocces.stast_df(df)

        MAX_NB_WORDS = len(stats_words['unique_words'])

        tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(df.Cleaned_text)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        X_train_cnn = pickle.load(open(self.path_models+'X_train_cnn.pickle', 'rb'))
        X_test_cnn = pickle.load(open(self.path_models+'X_test_cnn.pickle', 'rb'))
        Y_train_cnn = pickle.load(open(self.path_models+'Y_train_cnn.pickle', 'rb'))
        Y_test_cnn = pickle.load(open(self.path_models+'Y_test_cnn.pickle', 'rb'))

        logger.info('Shape of X_train: ' + str(X_train_cnn.shape))
        logger.info('Shape of X_test :' + str(X_test_cnn.shape))

        embeddings_index = {}
        f = open('../../glove/glove.6B.100d.txt', encoding='ISO-8859-1')
        for line in f:
            try:
                values = line.split()
                word = values[0]
                coefs = np.array(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                pass
        f.close()

        logger.info('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(
                                    len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False
        )

        pat = 5 
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
        self.model_checkpoint_cnn = ModelCheckpoint('../../models/model.h5', verbose=1, save_best_only=True)

        sequence_input = Input(shape=(None,), dtype="int64")
        embedded_sequences = embedding_layer(sequence_input)
        x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.MaxPooling1D(5)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Conv1D(128, 5, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        preds = layers.Dense(len(Y_train_cnn[0]), activation="softmax")(x)
        self.model_cnn = Model(sequence_input, preds)

        self.model_cnn.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc']
        )
        n_folds = 3
        epochs = 20
        batch_size = 128

        # save the model history in a list after fitting
        model_history_cnn = []

        for i in range(n_folds):
            print("Training on Fold: ", i+1)
            t_x, val_x, t_y, val_y = train_test_split(
                X_train_cnn, Y_train_cnn, test_size=0.1,
                random_state=np.random.randint(1, 1000, 1)[0])
            model_history_cnn.append(self.fit_and_evaluate_cnn(t_x, val_x, t_y, val_y, epochs, batch_size))
            print("======="*12, end="\n\n\n")
        
        logger.info('Training CNN model')
        logger.info("The model has been trained")
        logger.info("Saving model ")
        
        logger.info(f"Add this path for the next step classify: ../../models/model.h5")

        return True

    def fit_and_evaluate_cnn(self, t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=128):
        model = None
        model = self.model_cnn
        results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            callbacks=[self.early_stopping, self.model_checkpoint_cnn],
                            verbose=1, validation_split=0.1)
        print("Val Score: ", model.evaluate(val_x, val_y))
        return results


if __name__ == "__main__":
    logger.info(sys.argv[1])
    training_model = Train(sys.argv[1])

    training_model.train_model()
