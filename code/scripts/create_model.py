import pickle
import logging
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.preprocessing import LabelBinarizer
from itertools import cycle


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)


class PreproccesModel:
    def __init__(self, df, stats_words):
        self.df = df
        self.max_features = stats_words['unique_words']
        self.path_models = "../../models/"

    def label_encoder(self):
        """
        Load labelencoder
        
        Args:
            :: 

        Returns:
            :lb: Labelencoder object
        
        """

        try:
            lb = pickle.load(open(self.path_models+'label.pickle', 'rb'))
            return lb
        except Exception:
            lb = LabelBinarizer()
            label = lb.fit_transform(self.df.label.to_list())
            return label

    def tfidf_vector(self, min_df=None):
        """
        Load TFIDF Vector
        
        Args:
            :min_df:  Minimum of frecuency of documents to be consider

        Returns:
            :tfidf: tfidf object
        
        """

        try:
            tfidf = pickle.load(open(self.path_models+'tfidf.pickle', 'rb'))
            logger.info('Number of words to train the model:' + str(len(tfidf.get_feature_names())))
        except Exception:
            word_vec = TfidfVectorizer(
                analyzer='word',
                max_df=0.65,
                min_df=min_df,
                ngram_range=(1, 1),
                max_features=len(self.max_features)
            )

            tfidf = word_vec.fit(self.df.Cleaned_text)
            logger.info('Number of words to train the model:' + str(len(word_vec.get_feature_names())))

        return tfidf
    
    def test_data(self):
        """
        Load test data to test the models
        
        Args:
            ::  

        Returns:
            :X_test: test data to test the model
            :Y_test: Label of test data
        
        """

        X_test = pickle.load(open(self.path_models + 'X_test_cnn.pickle', 'rb'))
        Y_test = pickle.load(open(self.path_models + 'Y_test_cnn.pickle', 'rb'))
        logger.info('Shape of test set ' + str(X_test.shape))
        return X_test, Y_test

    def load_cnn(self):
        """
        Load cnn model
        
        Args:
            ::  
        Returns:
            :mlp: cnn model
        
        """
        cnn = pickle.load(open(self.path_models+'model_cnn.pickle', 'rb'))
        return cnn

    def plot_loss(self):
        """
        Plot of loss and val loss function for LSTM
        
        """

        history = pickle.load(open(self.path_models+'history.pickle', 'rb'))
        plt.title('Loss')
        plt.plot(history['loss'], label='train')
        plt.plot(history['val_loss'], label='test')
        plt.legend()
        plt.show()
        return

    def plot_precision_recall_curve(self, y_predict, Y_test):
        """
        Plot presision vs recall curve to evaulate the models
        :param y_predict: prediction of the model
        :param Y_test: grand true
        :return:
        """

        lb = LabelBinarizer()

        y_predict_label = lb.fit_transform(y_predict)
        Y_test_label = lb.fit_transform(Y_test)

        n_clases = Y_test_label.shape[1]

        precision = dict()
        recall = dict()
        average_precision = dict()
        
        lines = []
        labels = []
        for i in range(n_clases):
            precision[i], recall[i], _ = precision_recall_curve(Y_test_label[:, i],
                                                                y_predict_label[:, i])
            average_precision[i] = average_precision_score(Y_test_label[:, i],
                                                           y_predict_label[:, i])

        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        for i, color in zip(range(n_clases), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, bbox_to_anchor=(1.04, 1), borderaxespad=0, prop=dict(size=14))
        return
