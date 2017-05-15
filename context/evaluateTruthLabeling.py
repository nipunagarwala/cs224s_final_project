import pickle
import pandas as pd
import numpy as np
import scipy

""" 
Samples k instances each of audible, whispered, and silent
data from the training data, and splits those files 
according to the estimated labels provided by the corpus 
authors for both words and phones, so that the timestamp->label 
quality can be estimated on the resulting split data files.
"""

fs = 16000 # known

with open("data/train/utteranceInfoTrain.pkl", "rb") as f:
    info = pickle.load(f)

    
path = "data/train"

totalSamplesPerMode = 5
for level in ["word", "phone"]:
    for mode in ["audible", "whispered", "silent"]:
        sampledFilenames = info[info["mode"] == mode].sample(n=totalSamplesPerMode, 
                    replace=False, random_state=1)[["transcript","label"]]
                    
        print("\n", level, mode)
        for i in sampledFilenames["transcript"]:
            print(i)
        for i in sampledFilenames["label"]:
            print(i)                    
            
        for fn in sampledFilenames["label"] + ".pkl":
            with open("%s/%s" % (path, fn), "rb") as f:
                audio, emg = pickle.load(f)
                
            audio[level+'Block'] = (audio[level].shift(1) != audio[level]).astype(int).cumsum()
            groupedBy = audio.reset_index().groupby([level+'Block',level])['index'].apply(np.array)
            
            if level == "phone":
                print("\n", fn)
                
            for idx, label in groupedBy.index:
                if level == "phone":
                    print("%4d %s" % (idx, label))
                    
                rowIds = groupedBy[idx][label]
                data = audio["audio_amplitude"][rowIds]
                scaled = np.int16(data/np.max(np.abs(data)) * 32767) # to integer
                scipy.io.wavfile.write("context/%s/%s/%s_%d_%s.wav" % (level, mode, fn, idx, label.replace("?", "unk")) , fs, scaled)