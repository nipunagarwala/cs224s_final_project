import pickle

import numpy as np


# Implements subsequences (selecting on changes in word-level data)

# Get some data to play with
#with open("data/dev/002_001_0317.pkl", "rb") as f:
with open("data/train/002_001_0100.pkl", "rb") as f:
    audio, emg = pickle.load(f)


def select_subsequence(emg):
    # Get locations of each word
    new_word_begins = np.hstack([[0], np.where(emg["word"][1:] != emg["word"][:-1])[0] + 1])
    #print(emg["word"][new_word_begins])
    
    # Select a random subsequence
    end_word, start_word = -1, -1
    while (end_word <= start_word or 
           end_word-start_word < 2 or 
           end_word-start_word < 3 and (start_word == 0 or end_word == len(new_word_begins)-1)):
        # Until we start before we begin, and we have a length of 2 -- or 3 if we are using $ words
        start_word = np.random.randint(len(new_word_begins)-2)
        end_word = np.random.randint(start_word+1, len(new_word_begins))
        
    start_loc = new_word_begins[start_word]
    end_loc = new_word_begins[end_word]
    
    transcript = " ".join(emg["word"][new_word_begins][start_word:end_word]).replace("$", "").strip()
    emg = emg[start_loc:end_loc]
    return emg, transcript

 
str = []
for _ in range(100):
    e, t = select_subsequence(emg)
    str.append(t)

str.sort()
for i in str:
    print(i)
    
    