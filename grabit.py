import requests
import json
import pymongo
import os
from datetime import datetime
from save import json_to_mongo

# get top 25 posts from reddit and save to mongo db
reddit_url = "https://www.reddit.com/r/all/hot.json?limit=100"
reddit_headers = {'User-agent' : 'sebis reddit bot 0.0.1'}
r = requests.get(url = reddit_url, headers = reddit_headers)
reddit_data = r.json()
now = datetime.now().isoformat()
reddit_data['date'] = now
reddit_data['_id'] = "reddit_data_dump_from_{0}".format(now)

json_to_mongo("top_25_dump", reddit_data)

for child in reddit_data['data']['children']:
	child_data = child['data']
	child_data['processed_date'] = datetime.now().isoformat()
	json_to_mongo("reddit_posts_meta_data", child_data)
