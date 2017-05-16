import os
import pickle

"""
Moves all the data *.pkl files out of data/* and into 
data/train/* and data/test/* according to whether the .pkl
file reflects a train or a test instance.
Creates utteranceInfoTest.pkl and utteranceInfoTrain.pkl.
"""

with open("data/utteranceInfo.pkl", "rb") as f:
    meta = pickle.load(f)
    
directory = "data"
for i, utterance in meta.iterrows():
    old_pkl_filename = os.path.join(directory, utterance["label"] + ".pkl")
    if utterance["split"] == "test":
        new_pkl_filename = os.path.join(directory, "test", utterance["label"] + ".pkl")
        os.rename(old_pkl_filename, new_pkl_filename)
    elif utterance["split"] == "train":
        new_pkl_filename = os.path.join(directory, "train", utterance["label"] + ".pkl")
        os.rename(old_pkl_filename, new_pkl_filename)
    else:
        raise ValueError("Unknown split: %s" % utterance["split"])
        

trainOnly = meta[meta["split"] == "train"]
testOnly = meta[meta["split"] == "test"]

with open(os.path.join(directory, "train", "utteranceInfoTrain.pkl"), "wb") as f: 
    pickle.dump(trainOnly, f)
    
with open(os.path.join(directory, "test", "utteranceInfoTest.pkl"), "wb") as f: 
    pickle.dump(testOnly, f)