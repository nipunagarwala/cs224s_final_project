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

        `speakerId, sessionId, utteranceId, label, transcript, gender, mode, train/test split`
		
To recover the audio & EMG data for 
a particular file, unpickle it from `$label.pkl`.
The `$label.pkl` files contain a tuple in which 
the audio data is the first element and 
the EMG data is the second element.

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
		  that the corpus authors guess was being uttered)