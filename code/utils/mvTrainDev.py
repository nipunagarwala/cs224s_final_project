import os
import pickle
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

"""
One-time use.

Splits data/train/* into data/train/* and data/dev/*
at random such that transcripts do not overlap between sets, 
with an 80-20 split.
As part of this, this script revises utteranceInfo.pkl
and creates utteranceInfo.pkl.
"""

with open("data/train/utteranceInfo.pkl", "rb") as f:
    meta = pickle.load(f)
    
PROB_OF_TRAINING = 0.8



# Transcripts can be used a variable number of times
# Iterate over each transcript and the number of times it is used,
# and split it and only its instances fairly among train/dev
#
# What is the distribution of transcript usage in their training dat?
#>>> meta["transcript"].value_counts().value_counts()
# 1     391
# 3       5
# 4      35
# 5       2
# 6      38
# 16      4
# 17     36
# aka 391 transcripts are used only 1 time, 5 transcripts are used 3 times, ...
transcript_uses = meta["transcript"].value_counts()
num_uses = transcript_uses.value_counts().index

train = []
dev = []
for use_count in num_uses:
    transcript_values = transcript_uses[transcript_uses == use_count].index
    t_transcripts, d_transcripts = train_test_split(transcript_values, train_size=PROB_OF_TRAINING, random_state=42)
    train.append( meta[meta["transcript"].isin(t_transcripts)] )
    dev.append( meta[meta["transcript"].isin(d_transcripts)] )
train = pd.concat(train)
dev = pd.concat(dev)
    
dev["split"] = "dev"

# Move around the results
directory = "data/train"
for i, utterance in train.iterrows():
    old_pkl_filename = os.path.join(directory, utterance["label"] + ".pkl")
    assert( utterance["split"] == "train" )
    new_pkl_filename = os.path.join(directory, "newtrain")#, utterance["label"] + ".pkl")
    shutil.copy(old_pkl_filename, new_pkl_filename)
for i, utterance in dev.iterrows():
    old_pkl_filename = os.path.join(directory, utterance["label"] + ".pkl")
    assert( utterance["split"] == "dev" )
    new_pkl_filename = os.path.join(directory, "newdev")#, utterance["label"] + ".pkl")
    shutil.copy(old_pkl_filename, new_pkl_filename)


# Write out the meta info 
with open(os.path.join(directory, "newtrain", "utteranceInfo.pkl"), "wb") as f: 
    pickle.dump(train, f)
    
with open(os.path.join(directory, "newdev", "utteranceInfo.pkl"), "wb") as f: 
    pickle.dump(dev, f)
    
    
    
##############################
# Move things around
##############################
os.rename("data/train/newtrain", "data/newtrain")
os.rename("data/train/newdev", "data/dev")

shutil.rmtree("data/train")

os.rename("data/newtrain", "data/train")
