EMG Data
========

The original data is available at http://csl.anthropomatik.kit.edu/EMG-UKA-Corpus.php. The original data is audio format.

For this project, the complete processed-as-numeric EMG and audio data is available through Google Drive. A sample of the data is available in the `sample-data` folder for testing and network memorization purposes.

From the raw data, we perform feature extraction.  Files for this preprocessing appear in the `code` folder.

Format
------

The data comes in two parts:
* `utteranceInfo.pkl` -- a Python 3 pickle file that contains all the meta information for each segment
* Files in the format `(\d\d\d)_(\d\d\d)_(\d*).pkl` -- Python 3 pickle files whose names reflect the original data segment names

See `sample-data` for example content.

The data have been processed to include only
the actual data (no additional synchronization
channels, & first and last 0.2 seconds are trimmed as 
per the offset values provided by the corpus 
authors).  The frameIds reflect 10 ms frames
as provided by the corpus authors; the frames are
the values that align the audio & EMG data.
The phone and word columns reflect the corpus authors' guesses as to phones and words being uttered during 
each frame.

### Overview format
The `utteranceInfo.pkl` file contains the metainformation (8 columns):
* speakerId, sessionId, utteranceId, label, transcript, gender, mode, train/test split

The file `utteranceInfoSample.pkl` for the sample data can be accessed as follows:

    >>> import pickle
    >>> import pandas
    >>> with open("utteranceInfo.pkl", "rb") as f:
    >>>     info = pickle.load(f)
    >>> info
	  speakerId sessionId utteranceId         label  \
	0       002       001        0311  002_001_0311   
	1       002       001        0312  002_001_0312   
	2       002       001        0313  002_001_0313   
	6       002       001        0317  002_001_0317   
	7       002       001        0318  002_001_0318   
       transcript                                         gender  mode   split  
	0  BOTH OUR CORPORATE AND FOUNDATION CONTRIBUTION...   male  silent  train  
	1  THE COALITIONS ARE SPONSORING TWO SEPARATE INI...   male  silent  train  
	2  THE RESULTS HAVE PROMPTED THEM TO QUESTION THE...   male  silent  train  
	6                                       WE CAN DO IT   male  silent  train  
	7  FINGERPRINTS WERE EXTENSIVELY USED AND ALSO DE...   male  silent  train
		
To recover the audio & EMG data for 
a particular file, unpickle it from `$label.pkl`.
The `$label.pkl` files contain a tuple in which 
the audio data is the first element and 
the EMG data is the second element.

    >>> import pickle
    >>> import pandas
    >>> with open("002_001_0311.pkl", "rb") as f:
    >>>     audio, emg = pickle.load(f)


### Audio data format
The audio data (16000 Hertz) has an index plus 5 columns:

* index (int sample ID from 0 to length of trimmed segment)

* audio_amplitude (signed float)

* rawSampleId (int from startOffset to endOffset -- the
			non-trimmmed consecutive sample IDs)

* frameId (int from 0 where 0 is used for the first 10 ms, 
		 1 is used for the second 10 ms, ...;
		 the audio data has a 16000 Hertz frame rate, so
		 each complete frameId recurs 160 times)
		 
* phone (string indicating corpus authors' guess as to 
	   phone -- for audible & whispered, the guess uses
	   forced alignments; for silent, the guess uses a 
	   model that matches)
	   
* word (string indicating the word from the transcript
	  that the corpus authors guess was being uttered)

The raw audio data is amplitude data.

    >>> audio.shape
    (99239, 5)
    >>> audio[0:5]
       audio_amplitude  rawSampleId  frameId phone word
    0        -0.000244         3672        0   SIL    $
    1        -0.000214         3673        0   SIL    $
    2        -0.000031         3674        0   SIL    $
    3         0.000122         3675        0   SIL    $
    4        -0.000061         3676        0   SIL    $

Because of the higher sampling frequency, there are more rows of audio than of EMG data.
		  
### EMG data

The EMG data (600 Hertz) has an index plus 10 columns:

* index (int sample ID from 0 to length of trimmed segment)

* emg1 (signed int; anterior belly of the digastric & tongue,
		derived unipolarly,
		+ reference electrode on nose)
		
* emg2 (signed int; levator anguli oris & zygomaticus major,
		derived bipolarly)  
		
* emg3 (signed int; levator anguli oris & zygomaticus major,
		derived unipolarly,
		+ reference electrode behind the ears)
		
* emg4 (signed int; platysma,
		derived unipolarly,
		+ reference electrode behind the ears)
		
* emg5 (signed int; removed in corpus authors' experiments 
		because it tends to yield unstable and
		artifact-prone signals;
		platysma & depressor anguli oris,
		derived unipolarly,
		+ reference electrode behind the ears)
		
* emg6 (signed int; tongue,
		derived bipolarly)
		
* rawSampleId (int from startOffset to endOffset -- the
			non-trimmmed consecutive sample IDs)
			
* frameId (int from 0 where 0 is used for the first 10 ms, 
		 1 is used for the second 10 ms, ...;
		 the audio data has a 600 Hertz frame rate, so
		 each complete frameId recurs 6 times)
		 
* phone (string indicating corpus authors' guess as to 
		 phone -- for audible & whispered, the guess uses
		 forced alignments; for silent, the guess uses a 
		 model that matches)
		 
* word (string indicating the word from the transcript
	  that is the corpus authors' guess as to what was being 
      uttered at that sample)

We can access EMG data as above, unpacking it as follows:

    >>> emg.shape
    (3720, 10)
    >>> emg[0:5]
       emg1  emg2  emg3  emg4   emg5  emg6  rawSampleId  frameId phone word
    0 -1699 -8119  3133  7616  11609  3431          204        0   SIL    $
    1 -1527 -8080  3050  7599  11503  3196          205        0   SIL    $
    2 -1561 -7903  3893  8324  12135  3337          206        0   SIL    $
    3 -1589 -8015  3132  7859  12106  3201          207        0   SIL    $
    4 -1661 -7817  3093  7646  11964  2772          208        0   SIL    $

Because of the lower sampling frequency, there are fewer rows of EMG than of audio data.

Quality of Annotations
------
The corpus authors provide estimates of the phone and word being uttered during each frame.  These are based on HMM alignment models on the audio track for the audible and whispered data, and those models plus another mapping for the silent data.  The corpus authors indicate that the alignments have not been explored by a human, and they make no representations as to quality, especially regarding the silent alignments.

In our quest to know the dataset with which we're working, we would like to know the quality of the alignments for the following reasons: (1) If the quality is poor, this supports moving to a CTC framework that does not require correct alignments, (2) If the quality is excellent (even in particular modes, or for words and not phones), we can use this to inform data augmentation efforts.

To understand the quality of those labels, I perform a hand analysis: by randomly selecting five audio utterances from each mode, I rate sample on a 5-point holistic scale as to the quality of its labels, and then average those scores.  For the silent data, which does not have an intelligible audio track, this analysis is facilitated through tells like plosives and length of segment.  The analysis is intended to provide signposts for the approximate quality of the data, rather than be an exhaustive qualitative study. 

    Overall Quality (Total sub-segments)
            audible     whispered     silent
    word    4.6  (62)   3.8  (65)     3.4  (57)
    phone   3.6 (194)   0.8  (194)    0.2 (188)

From this analysis, we learn that:
* We can trust the audible word-level data.  
* We can somewhat trust the word-level data of whispered segments & silent segments, and the audible phone alignments.  These data have some mistaggings, but is on the whole understandable.  
* It is inappropriate to use the phone-level information from whispered or silent speech.

In the above, I've rated each sentence holistically according to its sub-segments on the following scale:

* 5: excellent (perfect)
* 4: good (mostly perfect, 1 or 2 errors)
* 3: fair (has repeated mistaggings but is understandable)
* 2: poor (medium-length subsegments are misstagged)
* 1: awful (not understandable; long-length segments are entirely misstagged)
* 0: zilch (almost entirely garbage -- any correctness seems random)

Qualitative examples are available on Google Drive and in context/sampled.txt.


Splits
------
The data comes in 3 splits. The corpus authors provide a train/test split for compatibility with their reported results.  We split the training data further into a train/dev split (80-20).  Each transcript only occurs in a single split, though it might repeat within that split across speakers, modes, sessions, or within the same speaker/mode/session combination.  The final splits are:

* Train:
    * 1145 utterances (in ratio of 4-1-1: 771 audible, 187 whispered, 187 silent)
    * 406 unique transcripts (312 repeated once, 28 repeated 17 times, with repetition values in between)
    * 4 speakers (in data ratio of 6-1-1-4)
    * 91.8% male
* Dev: 
    * 315 utterances (in ratio of 4-1-1: 209 audible, 53 whispered, 53 silent)
    * 105 unique transcripts (79 repeated once, 8 repeated 17 times, with repetition values in between)
    * 4 speakers (in data ratio of 6-1-1-4)
    * 91.4% male
* Test:
    * 260 utterances (in ratio of 2-1-1: 140 audible, 60 whispered, 60 silent)
    * 10 unique transcripts (26 repeated 10 times)
    * 4 speakers (in data ratio of 3-1-1-4)
    * 88.5% male

Each train/dev/test split is a subdirectory of `data` folder, and each has its own `utteranceInfo.pkl` file. 