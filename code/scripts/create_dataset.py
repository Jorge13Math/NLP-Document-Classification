import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)

def genererate_dataframe(path):
    """
    Generate dataframe from folder data
    
    Args:
        :path: Folder's Path

    Returns:
        :df: Dataframe with text label and language
    
    """
    logger.info('Loading files')
    labels = os.listdir(path)
    values = {}
    data = []
    for label in labels:
        if os.path.isdir(path+label):
            for doc in os.listdir(path+label):
                f = open(path+label+'/'+doc,'r',encoding='ISO-8859-1')
                text = f.read()
                values['Text']=text
                f.close()
                values['label']= label
                values['documentName'] = doc
                data.append(values)
                values ={}
    
    df = pd.DataFrame(data)
    logger.info('Dataframe generated')
   
    return df