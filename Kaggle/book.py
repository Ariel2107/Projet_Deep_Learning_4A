import pandas as pd
import time
import datetime
import mgzip
import pickle

with mgzip.open('./data/cleaned_ds.gz', 'rb') as f:
    df_reviews = pickle.load(f)

time = time.time()

print(df_reviews)#bon

df_reviews_mean = df_reviews.groupby(['book_id']).mean()['rating']
print(df_reviews_mean)#bon

df_reviews_duration= pd.to_datetime(df_reviews['date_updated']) - pd.to_datetime(df_reviews['started_at'])
print(df_reviews_duration)# temps moyen par livre pour chaque utilisateur

df_reviews['duration'] = df_reviews_duration

df_reviews_mean_duration = df_reviews.groupby(['book_id']).mean(numeric_only=False)['duration']
print(df_reviews_mean_duration)#bon

df_reviews['duration_mean'] = df_reviews_mean_duration

df_reviews_nb_comments= df_reviews.groupby(['book_id']).sum()['n_comments']
print(df_reviews_nb_comments)#bon

df_reviews_std_rating = df_reviews.groupby(['book_id']).std()['rating']
print(df_reviews_std_rating)#bon controverse

