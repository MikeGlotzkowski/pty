import json
import os
import pymongo
from pathlib import Path

def json_to_file(file_name, json_data):
	json_data_dump = json.dumps(json_data)
	save_path = "./save/{0}.json".format(file_name)
	Path("./save").mkdir(parents=True, exist_ok=True)
	save_file = open(save_path, "w")
	save_file.write(json_data_dump)
	save_file.close()

def json_to_mongo(container_name, json_data):
	username = os.environ.get("MONGO_USERNAME")
	password = os.environ.get("MONGO_PASSWORD")
	mongo_client = pymongo.MongoClient("mongodb://{0}:{1}@localhost:27017/".format(username, password))
	mongo_database = mongo_client["reddit_data"]
	mongo_container = mongo_database[container_name]
	mongo_container.insert(json_data)
	mongo_client.close()
