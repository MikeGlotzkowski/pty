import pandas as pd
import os
from pymongo import MongoClient
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# read mongo db collection data into dataframe
username = os.environ.get("MONGO_USERNAME")
password = os.environ.get("MONGO_PASSWORD")
mongo_client = MongoClient("mongodb://{0}:{1}@localhost:27017/".format(username, password))
cursor = mongo_client["reddit_data"]["reddit_posts_meta_data"].find({}) # finds everything

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
'crosspost_parent_list'], axis=1, inplace=True)

print(reddit_data_df.head())

# feature_hasher = FeatureHasher(n_features = 10)
#feature_hasher = DictVectorizer(sparse=False)
feature_hasher = LabelEncoder()
reddit_data_df['subreddit'] = feature_hasher.fit_transform(reddit_data_df['subreddit'])

print(reddit_data_df.head())



# randomly split test/train
msk = np.random.rand(len(reddit_data_df)) < 0.8
train_x = reddit_data_df[msk]
test_x = reddit_data_df[~msk]
train_y = train_x.pop('score')
test_y = test_x.pop('score')

model = XGBClassifier()
model.fit(train_x, train_y)
