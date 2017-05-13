"""
@author Pamela Toman
"""

import os
import re
import pickle
from glob import glob

import numpy as np
import pandas as pd
import soundfile as sf

import argparse

def write_to_disk(pathnameToCorpus):
    # Get the data divisions into train/test
    def populateVariable(var, fn):
        with open(os.path.join(pathnameToCorpus, "Subsets", fn)) as f:
            for line in f:
                _, fileString = line.strip().split(":")
                var.extend(fileString.split())

    test_audible, test_silent, test_whispered = [], [], []
    populateVariable(test_audible, "test.audible")
    populateVariable(test_silent, "test.silent")
    populateVariable(test_whispered, "test.whispered")

    train_audible, train_silent, train_whispered = [], [], []
    populateVariable(train_audible, "train.audible")
    populateVariable(train_silent, "train.silent")
    populateVariable(train_whispered, "train.whispered")
            
            
    # Figure out which files to parse
    files = glob(os.path.join(pathnameToCorpus, "audio", "*", "*", "*"))
    files = [os.path.basename(f) for f in files]
    to_parse = []
    for f in files:
        m = re.match(".*\_(\d\d\d)\_(\d\d\d)\_(\d\d\d\d)\.wav", f)
        speaker = m.group(1)
        session = m.group(2)
        utterance = m.group(3)
        to_parse.append((speaker, session, utterance))

    # Units to keep and reformat into a dataset
    labels = []
    transcripts = []
    genders = []
    modes = []
    subsets = []

    # Create dataset
    for idx, (speaker, session, utterance) in enumerate(to_parse):
        label = "%s_%s_%s" % (speaker, session, utterance)
        
        print(idx, label)
        
        # SAMPLE LEVEL
        # Audio data
        audioData, audioFs = sf.read(pathnameToCorpus + "audio/%s/%s/a_%s.wav" % (speaker, session, label))
        assert(audioFs == 16000) # Hertz

        # EMG data
        emgFs = 600 # Hertz
        with open(pathnameToCorpus + "emg/%s/%s/e07_%s.adc" % (speaker, session, label), "rb") as f:    
            DATA_TYPE = '<i2'   # little endian 2 byte integer
            N_CHANNELS = 7      # 7 EMG channels
            emgData = np.fromfile(f, dtype=np.dtype(DATA_TYPE)) 
            emgData = emgData.reshape(-1, N_CHANNELS)
            # convert to int32 to match sample MATLAB code
            emgData = emgData.astype("int32") 
            
        # Offset data
        with open(pathnameToCorpus + "offset/%s/%s/offset_%s.txt" % (speaker, session, label), "r") as f:   
            lines = f.readlines()
            if (len(lines) != 2):
                raise ValueError("Offset file %s should contain exactly 2 lines" % label)
            audioStart, audioEnd = [int(elem) for elem in lines[0].strip().split()]
            emgStart, emgEnd = [int(elem) for elem in lines[1].strip().split()]

        # Cut off the edges to get them aligned  
        ad = audioData[audioStart:audioEnd,:]
        emgData = emgData[emgStart:emgEnd,:]

        # Add frame IDs to both using 10 ms frames
        ad = pd.DataFrame(data=ad[:,0], columns=["audio_amplitude"])
        ad["rawSampleId"] = range(audioStart, audioEnd)
        ad["frameId"] = np.floor(ad.index / (audioFs/100)).astype(np.int32)
        ad["phone"] = ["?"]*len(ad)
        ad["word"] = ["?"]*len(ad)

        ed = pd.DataFrame(data=emgData[:,0:6], columns=["emg1", "emg2", "emg3", "emg4", "emg5", "emg6"])
        ed["rawSampleId"] = range(emgStart, emgEnd)
        ed["frameId"] = np.floor(ed.index / (emgFs/100)).astype(np.int32)
        ed["phone"] = ["?"]*len(ed)
        ed["word"] = ["?"]*len(ed)

        # Assign the phone alignments
        # (these are probably bad)
        with open(pathnameToCorpus + "Alignments/%s/%s/phones_%s.txt" % (speaker, session, label), "r") as f:
            for line in f:
                start, end, phone = line.strip().split()
                for i in range(int(start), int(end) + 1):
                    ed.loc[ed["frameId"] == i, "phone"] = phone
                    ad.loc[ad["frameId"] == i, "phone"] = phone

        # Assign the word alignments
        # (these are probably bad)
        with open(pathnameToCorpus + "Alignments/%s/%s/words_%s.txt" % (speaker, session, label), "r") as f:
            for line in f:
                start, end, word = line.strip().split()
                for i in range(int(start), int(end) + 1):
                    ed.loc[ed["frameId"] == i, "word"] = word
                    ad.loc[ad["frameId"] == i, "word"] = word

        # SEGMENT LEVEL
        # Get full text transcript
        with open(pathnameToCorpus + "Transcripts/%s/%s/transcript_%s.txt" % (speaker, session, label), "r") as f:
            lines = f.readlines()
            if (len(lines) != 1):
                raise ValueError("Transcript file %s should contain exactly 1 line" % label)
            transcript = lines[0].strip()

        # Gender
        if speaker in ["002", "006", "008"]:
            gender = "male"
        elif speaker in ["004"]:
            gender = "female"
        else:
            raise ValueError("Speaker %s must be in known set of speakers" % speaker)
        
        # Mode
        if speaker == "002" and session in ["001", "003"]:
            # Use 1-indexing
            m = re.match("\d(\d)\d\d", utterance)
            raw_mode = m.group(1)
            if raw_mode == "1":
                mode = "audible"
            elif raw_mode == "2":
                mode = "whispered"
            elif raw_mode == "3":
                mode = "silent"
            else:
                raise ValueError("Mode for %s must be in [audible|whispered|silent]" % label)
        elif speaker in ["004", "006", "008"] and session in ["001", "002", "003", "004", "005", "006", "007", "008"]:
            # Use 0 indexing
            m = re.match("\d(\d)\d\d", utterance)
            raw_mode = m.group(1)
            if raw_mode == "0":
                mode = "audible"
            elif raw_mode == "1":
                mode = "whispered"
            elif raw_mode == "2":
                mode = "silent"
            else:
                raise ValueError("Mode for %s must be in [audible|whispered|silent]" % label)
        elif speaker == "002" and session in ["101"]:
            # long session is always audible
            mode = "audible"
        else:
            raise ValueError("Mode unrecoverable")
        
        # Data split
        if ("emg_%s-%s-%s" % (speaker, session, utterance)) in test_audible:
            assert(mode == "audible")
            subset = "test"
        elif ("emg_%s-%s-%s" % (speaker, session, utterance)) in test_whispered:
            assert(mode == "whispered")
            subset = "test"
        elif ("emg_%s-%s-%s" % (speaker, session, utterance)) in test_silent:
            assert(mode == "silent")
            subset = "test"
            
        elif ("emg_%s-%s-%s" % (speaker, session, utterance)) in train_audible:
            assert(mode == "audible")
            subset = "train"
        elif ("emg_%s-%s-%s" % (speaker, session, utterance)) in train_whispered:
            assert(mode == "whispered")
            subset = "train"
        elif ("emg_%s-%s-%s" % (speaker, session, utterance)) in train_silent:
            assert(mode == "silent")
            subset = "train"
            
        else:
            raise ValueError("Unknown split for %s" % label)
        
        #print()
        #print(transcript)
        #print(speaker, session, utterance)
        #print(label)
        #print(gender, mode)
        #print(subset)
        #print(ed[1001:1003])
        #print(ad[1001:1003])
        
        with open("%s.pkl" % label, "wb") as f:
            pickle.dump( (ad, ed), f)
        
        labels.append(label)
        transcripts.append(transcript)
        genders.append(gender)
        modes.append(mode)
        subsets.append(subset)

    # Create an overall utterance info object
    utteranceInfo = pd.DataFrame(data=to_parse,
        columns=["speakerId", "sessionId", "utteranceId"])
    utteranceInfo["label"] = labels
    utteranceInfo["transcript"] = transcripts
    utteranceInfo["gender"] = genders
    utteranceInfo["mode"] = modes
    utteranceInfo["split"] = subsets

    # Dump the processed data to disk so that we don't 
    # have to wait for all the processing in the future
    with open("utteranceInfo.pkl", "wb") as f:
        pickle.dump(utteranceInfo, f)
        
    return utteranceInfo

def load_data():
    with open("utteranceInfo.pkl", "rb") as f:
        utteranceInfo = pickle.load(f)
    return utteranceInfo
    
def get_data(pathnameToCorpus):
    """ 
    Returns the metainformation (8 columns):
        speakerId, sessionId, utteranceId, label, 
        transcript, gender, mode, train/test split`
            
    To recover the audio & EMG data for 
    a particular file, unpickle it from `$label.pkl`.
    The `$label.pkl` files contain a tuple in which 
    the audio data is the first element and 
    the EMG data is the second element.
    
    The data have been processed to include only
    the actual data (no additional synchronization
    channels, & first and last 0.2 seconds are trimmed as 
    per the offset values provided by the corpus 
    authors).  The frameIds reflect 10 ms frames
    as provided by the corpus authors; the frames are
    the values that align the audio & EMG data.
    The phone and word columns reflect the corpus authors'  
    guesses as to phones and words being uttered during 
    each frame.
    The audio data (16000 Hertz) has an index plus 5 columns:
        index (int sample ID from 0 to length of trimmed segment)
        audio_amplitude (signed float)
        rawSampleId (int from startOffset to endOffset -- the
                    non-trimmmed consecutive sample IDs)
        frameId (int from 0 where 0 is used for the first 10 ms, 
                 1 is used for the second 10 ms, ...;
                 the audio data has a 16000 Hertz frame rate, so
                 each complete frameId recurs 160 times)
        phone (string indicating corpus authors' guess as to 
               phone -- for audible & whispered, the guess uses
               forced alignments; for silent, the guess uses a 
               model that matches)
        word (string indicating the word from the transcript
              that the corpus authors guess was being uttered)
    The EMG data (600 Hertz) has an index plus 10 columns:
        index (int sample ID from 0 to length of trimmed segment)
        emg1 (signed int; anterior belly of the digastric & tongue,
                derived unipolarly,
                + reference electrode on nose)
        emg2 (signed int; levator anguli oris & zygomaticus major,
                derived bipolarly)  
        emg3 (signed int; levator anguli oris & zygomaticus major,
                derived unipolarly,
                + reference electrode behind the ears)
        emg4 (signed int; platysma,
                derived unipolarly,
                + reference electrode behind the ears)
        emg5 (signed int; removed in corpus authors' experiments 
                because it tends to yield unstable and
                artifact-prone signals;
                platysma & depressor anguli oris,
                derived unipolarly,
                + reference electrode behind the ears)
        emg6 (signed int; tongue,
                derived bipolarly)
        rawSampleId (int from startOffset to endOffset -- the
                    non-trimmmed consecutive sample IDs)
        frameId (int from 0 where 0 is used for the first 10 ms, 
                 1 is used for the second 10 ms, ...;
                 the audio data has a 600 Hertz frame rate, so
                 each complete frameId recurs 6 times)
        phone (string indicating corpus authors' guess as to 
                 phone -- for audible & whispered, the guess uses
                 forced alignments; for silent, the guess uses a 
                 model that matches)
        word (string indicating the word from the transcript
              that the corpus authors guess was being uttered)
    """
    if os.path.isfile("utteranceInfo.pkl"):
        utteranceInfo = load_data()
    else: 
        utteranceInfo = write_to_disk(pathnameToCorpus)
        
    EXPECTED_UTTERANCES = 1720
    assert(len(utteranceInfo) == EXPECTED_UTTERANCES)
    
    return utteranceInfo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts data from EKG-UKA corpus")
    parser.add_argument("pathnameToCorpus", type=str, description="Location of corpus")
    args = parser.parse_args()
    utteranceInfo = get_data(args.pathnameToCorpus)