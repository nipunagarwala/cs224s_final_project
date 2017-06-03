import os
import pickle

WORDS = set()

def words(text):
    "List of words in text."
    return text.split()

for dataset in ["train", "dev", "test"]:
	meta_info_path = os.path.join("data", dataset, "utteranceInfo.pkl")
	with open(meta_info_path, "rb") as f:
		meta = pickle.load(f)
		for i, utterance in meta.iterrows():
			for word in words(utterance["transcript"]):
				WORDS.add(word)

with open("data/lm/words.txt", "w") as f:
	for word in WORDS:
		print(word, file=f)