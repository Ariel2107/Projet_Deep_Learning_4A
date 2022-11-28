import re
import nltk  # Natural Language Toolkit
import mgzip
import pickle
import string
import contractions
import pandas as pd

from tqdm import tqdm
from os.path import exists
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# ################################# CONSTANTS ##################################


PUNC_TO_REMOVE = string.punctuation
LANG = 'english'
SS = SnowballStemmer(language=LANG)

# Train dataset from Kaggle
DS_PATH = './Kaggle/data/goodreads_train.csv'

# Path to save the dataset with the reviews already cleaned
ZIP_PATH = './Kaggle/data/cleanedReviews.gz'


# ################################# METHODS ##################################


def clean_text(sentence, all_stopwords):
    # Lowercase the sentence
    sentence = sentence.lower()

    # Remove links
    sentence = re.sub(r"(\w+://\S+)|^rt|http.+?", "", sentence)

    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', PUNC_TO_REMOVE))

    # Remove numbers
    sentence = ''.join([char for char in sentence if not char.isdigit()])

    # Remove emojis and emoticons
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    sentence = emoji_pattern.sub(r'', sentence)

    # Remove contractions
    sentence = contractions.fix(sentence)

    # Stemming and remove stopwords and extra spaces
    sentence = sentence.split()
    sentence = [SS.stem(word) for word in sentence if word not in all_stopwords]
    sentence = ' '.join(sentence)

    return sentence


def clean_data(column):
    nltk.download('stopwords')
    all_stopwords = stopwords.words(LANG)
    all_stopwords.remove('not')
    tqdm.pandas()

    new_column = column.progress_map(lambda x: clean_text(x, all_stopwords))

    return new_column


def main():
    if exists(ZIP_PATH):
        print('Dataset already cleaned, saved at %s' % ZIP_PATH)
    else:
        print('Opening original dataset')
        reviews_df = pd.read_csv(DS_PATH)
        print('Cleaning reviews')

        reviews_df['cleaned_review_text'] = clean_data(reviews_df['review_text'])

    print('Saving cleaned dataset at %s' % ZIP_PATH)
    with mgzip.open(ZIP_PATH, 'wb') as f:
        pickle.dump(reviews_df, f)