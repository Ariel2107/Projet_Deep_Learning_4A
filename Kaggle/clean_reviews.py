import re
import nltk #Natural Language Toolkit
import mgzip
import pickle
import pandas as pd

from tqdm import tqdm

from os.path import exists

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def cleanText(sentence, ps, stopwords):
    sentence = re.sub('[^a-zA-Z]', ' ', sentence) ## ATT
    sentence = sentence.lower()
    sentence = sentence.split()
    sentence = [ps.stem(word) for word in sentence if not word in stopwords] 
    sentence = ' '.join(sentence)
    return sentence

def main() :
    ################################## PATHS ##################################

    # Train dataset from Kaggle
    ds_path = 'Kaggle/data/goodreads_train.csv'

    # Path to save the dataset with the reviews already cleaned
    zip_path = 'Kaggle/data/Acleaned_ds.gz'

    
    if exists(zip_path):
        print('Dataset already cleaned, saved at %s' %zip_path)
    else:
        print('Opening original dataset')
        reviews_df = pd.read_csv(ds_path)
        print('Cleaning rewiews')
        
        ps = PorterStemmer()
        nltk.download('stopwords')
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        tqdm.pandas()
        
        reviews_df['cleaned_review_text'] = reviews_df['review_text'].progress_map(lambda x : cleanText(x, ps, all_stopwords))
        print('Saving cleaned dataset at %s' %zip_path)
        with mgzip.open(zip_path, 'wb') as f:
            pickle.dump(reviews_df, f)