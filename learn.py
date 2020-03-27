import pandas as pd
import os
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score

from sklearn.impute import SimpleImputer
import numpy as np
from feature_engineering import avg_word, clean_dataset
import nltk
nltk.download('stopwords')


# read mongo db collection data into dataframe
username = os.environ.get("MONGO_USERNAME")
password = os.environ.get("MONGO_PASSWORD")
mongo_client = MongoClient(
    "mongodb://{0}:{1}@localhost:27017/".format(username, password))
cursor = mongo_client["reddit_data"]["reddit_posts_meta_data"].find(
    {})  # finds everything

reddit_data_df = pd.DataFrame(list(cursor))
mongo_client.close()
# remove own properties
del reddit_data_df['_id']
del reddit_data_df['processed_date']

# drop unused columns

reddit_data_df.drop([
    'approved_at_utc',
    'selftext',
    'author_fullname',
    'mod_reason_title',
    'link_flair_richtext',
    'subreddit_name_prefixed',
    'link_flair_css_class',
    'link_flair_text_color',
    'author_flair_background_color',
    'author_flair_template_id',
    'secure_media',
    'secure_media_embed',
    'link_flair_text',
    'thumbnail',
    'edited',
    'author_flair_css_class',
    'link_flair_type',
    'removed_by_category',
    'banned_by',
    'author_flair_type',
    'selftext_html',
    'banned_at_utc',
    'preview',
    'author_flair_text',
    'removed_by',
    'mod_reason_by',
    'removal_reason',
    'link_flair_background_color',
    'id',
    'report_reasons',
    'whitelist_status',
    'mod_reports',
    'author_flair_text_color',
    'permalink',
    'parent_whitelist_status',
    'media',
    'link_flair_template_id',
    'crosspost_parent_list',
    'name',
    'media_embed',
    'user_reports',
    'category',
    'approved_by',
    'author_flair_richtext',
    'gildings',
    'post_hint',
    'mod_note',
    'likes',
    'view_count',
    'all_awardings',
    'awarders',
    'num_reports',
    'distinguished',
    'subreddit_id',
    'discussion_type',
    'url',
    'content_categories',
    'crosspost_parent'], axis=1, inplace=True)

labelEncoder = LabelEncoder()
reddit_data_df['subreddit'] = labelEncoder.fit_transform(
    reddit_data_df['subreddit'])
reddit_data_df['subreddit_type'] = labelEncoder.fit_transform(
    reddit_data_df['subreddit_type'])
reddit_data_df['domain'] = labelEncoder.fit_transform(reddit_data_df['domain'])
reddit_data_df.suggested_sort.fillna(value="none", inplace=True)
reddit_data_df['suggested_sort'] = labelEncoder.fit_transform(
    reddit_data_df['suggested_sort'])
reddit_data_df['author'] = labelEncoder.fit_transform(reddit_data_df['author'])

# https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
reddit_data_df['title_word_count'] = reddit_data_df['title'].apply(
    lambda x: len(str(x).split(" ")))
# this also includes spaces
reddit_data_df['title_char_count'] = reddit_data_df['title'].str.len()
reddit_data_df['title_avg_word'] = reddit_data_df['title'].apply(
    lambda x: avg_word(x))
stop = nltk.corpus.stopwords.words('english')
reddit_data_df['title_stopwords'] = reddit_data_df['title'].apply(
    lambda x: len([x for x in x.split() if x in stop]))
reddit_data_df['title_hastags'] = reddit_data_df['title'].apply(
    lambda x: len([x for x in x.split() if x.startswith('#')]))
reddit_data_df['title_numerics'] = reddit_data_df['title'].apply(
    lambda x: len([x for x in x.split() if x.isdigit()]))
reddit_data_df['title_upper'] = reddit_data_df['title'].apply(
    lambda x: len([x for x in x.split() if x.isupper()]))
reddit_data_df.drop(['title'], axis=1, inplace=True)

# print(np.any(np.isnan(reddit_data_df)))
# print(np.all(np.isfinite(reddit_data_df)))

reddit_data_df = reddit_data_df.reset_index()
reddit_data_df = clean_dataset(reddit_data_df)

# randomly split test/train
msk = np.random.rand(len(reddit_data_df)) < 0.8
train_X = reddit_data_df[msk]
# test_X = reddit_data_df[~msk]
train_y = train_X.pop('score')
# test_y = test_X.pop('score')

feature_number = 20
train_X = SelectKBest(
    f_regression, k=feature_number).fit_transform(train_X, train_y)
# test_X = SelectKBest(
#     f_regression, k=feature_number).fit_transform(test_X, test_y)

model = XGBClassifier()
model.fit(train_X, train_y)

model = XGBClassifier()
kfold = KFold(n_splits=10)
results = cross_val_score(model, train_X, train_y, cv=kfold, scoring="neg_mean_squared_error")

print(results)

# https://hackernoon.com/want-a-complete-guide-for-xgboost-model-in-python-using-scikit-learn-sc11f31bq
# predict_train = model.predict(train_X)
# mean_squared_error_train = mean_squared_error(train_y, predict_train)
# print('\nmean_squared_error on train dataset : ', mean_squared_error_train)

# predict_test = model.predict(test_X)
# mean_squared_error_test = mean_squared_error(test_y, predict_test)
# print('\nmean_squared_error on test dataset : ', mean_squared_error_test)
