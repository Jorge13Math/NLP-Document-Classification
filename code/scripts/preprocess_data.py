import pandas as pd
import re
import matplotlib.pyplot as plt
import logging
import gensim
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS as en_stop


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)

stopwords = list(en_stop)


class Preprocces:
    """Class to Preprocess data"""

    def __init__(self, data):
        self.data = data
    
    def clean_dataframe(self):
        """Clean dataframe"""

        df = self.data

        logger.info('Shape of dataframe:' + str(df.shape))
        logger.info('Checking is there is null values')

        if len(df[df.isna().any(axis=1)]):
            logger.info('Remove null values: ' + str(len(df[df.isna().any(axis=1)])))
            df.dropna(inplace=True)
        logger.info('Checking duplicates values')
        if len(df[df.duplicated(subset=['Text'])]) > 1:
            logger.info('Remove duplicates values: ' + str(len(df[df.duplicated(subset=['Text'])])))
            df.drop_duplicates(subset=['Text'], inplace=True)
            
        logger.info('Clean text')
        logger.info('Remove digits and digits with words example : 460bc --> ""')
        logger.info('Transform words to lowercase example: Electronics --> electronics')
        logger.info('Remove special characters example : $# -->"" ')
        logger.info('Remove words with lenght less than three example : for --> "" ')
        logger.info('Remove stop words example : the --> "" ')
        df['Cleaned_text'] = df['Text'].apply(self.clean_text)
        
        df = df[df['Cleaned_text'] != '']
        
        df['number_words'] = df['Cleaned_text'].apply(lambda x: len(x.split()))
        df['number_unique_words'] = df['Cleaned_text'].apply(lambda x: len(set(x.split())))
        df = df.reset_index(drop=True)
        logger.info('Dataframe is cleaned')
        logger.info('Shape of dataframe:' + str(df.shape))
        return df

    def clean_text(self, text):
        """"Clean text from dataframe"""

        # 'Transform words to lowercase example: Electronics --> electronics'
        text = text.lower()
        
        # Remove digits and digits with words example : 460bc --> ""
        text = re.sub(r"\w*\d\w*", '', text)
        
        # Remove email: jorge@alberto: --> ""
        text = re.sub(r"\S*@\S*\s?", '', text)
        
        # Remove special characters example : $# -->""
        text = " ".join(gensim.utils.simple_preprocess(text))

        # Remove words with lenght less than three example : for --> "" 
        text = self.remove_words_l3(text)

        # Remove stop words in any language example : para --> ""
        pattern = re.compile(r"\b(' + r'|'.join(stopwords) + r')\b\s*")
        text = pattern.sub('', text)

        return text

    def remove_words_l3(self, text):
        """
        Remove words with length less than 4
        :param text: text to clean
        :return: text clean
        """

        token_text = text.split()
        
        clean_text = " ".join([word for word in token_text if len(word) > 3])
            
        return clean_text

    def stast_df(self, df):

        unique_words = set()
        df['Cleaned_text'].str.split().apply(unique_words.update)

        logger.info('Total of Unique words in text :' + str(len(unique_words)))

        count_words = Counter()
        df['Cleaned_text'].str.split().apply(count_words.update)

        values = count_words.values()

        total = sum(values)
        logger.info('Total of words in text :' + str(total))
        stats_data = {'unique_words': unique_words, 'count_words': count_words}
    
        return stats_data

    def plot_categories(self, df):
        df_category = pd.DataFrame({'Category': df.label.value_counts().index,
                                    'Number_of_documents': df.label.value_counts().values})
        df_category.plot(x='Category', y='Number_of_documents', kind='bar', legend=False, grid=True, figsize=(8, 5))
        plt.title("Number of documents per category")
        plt.ylabel('# of Documents', fontsize=12)
        plt.xlabel('Category', fontsize=12)

        return 

    def plot_common_words(self, count_words):
        sort_words = sorted(count_words.items(), key=lambda x: x[1], reverse=True)
        data = sort_words[:20]
        n_groups = len(data)
        values = [x[1] for x in data]
        words = [x[0] for x in data]
        plt.figure(figsize=(8, 5))
        plt.xticks(rotation=90)
        plt.bar(range(n_groups), values, tick_label=words)
        plt.title("Twenty most common words")
        plt.ylabel('# Ocurrences', fontsize=12)
        plt.xlabel('Word', fontsize=12)
        return 
    
    def plot_less_common_words(self, count_words):
        sort_words = sorted(count_words.items(), key=lambda x: x[1], reverse=False)
        data = sort_words[:20]
        n_groups = len(data)
        values = [x[1] for x in data]
        words = [x[0] for x in data]
        plt.figure(figsize=(8, 5))
        plt.xticks(rotation=90)
        plt.bar(range(n_groups), values, tick_label=words)
        plt.title("Twenty less common words")
        plt.ylabel('# Ocurrences', fontsize=12)
        plt.xlabel('Word', fontsize=12)
        return
