import pickle
import os

"""
One-time use.

Checks whether transcripts overlap between train & test,

and checks whether transcripts overlap between our train & our dev.
"""

with open("data/train/newtrain/utteranceInfo.pkl", "rb") as f:
    our_train = pickle.load(f)

with open("data/train/newdev/utteranceInfo.pkl", "rb") as f:
    dev = pickle.load(f)

with open("data/train/utteranceInfo.pkl", "rb") as f:
    train = pickle.load(f)

with open("data/test/utteranceInfo.pkl", "rb") as f:
    test = pickle.load(f)

    
d = set(dev["transcript"].unique())
tr_ours = set(our_train["transcript"].unique())
tr_all = set(train["transcript"].unique())
te = set(test["transcript"].unique())


len(te) # 10
len(tr_all) # 511
len(tr_ours) # 427
len(d) # 171

# Any overlap between their train & test?  No.
len(te & tr_all)  # 0

# Any overlap between our train and our dev? No.
len(tr_ours & d)  # 0


##############################
# Get statistics
##############################

our_train["mode"].value_counts()
#audible      771   4.1*lowest
#whispered    187
#silent       187
our_train["gender"].value_counts()
#male      1052     11.3*lowest
#female      93
our_train["transcript"].value_counts().value_counts()
#1     312
#3       4
#4      28
#5       1
#6      30
#16      3
#17     28
our_train["speakerId"].value_counts()
#002    584     6.28* lowest
#004     93     1* lowest
#006     93     1* lowest
#008    375     4* lowest
our_train.groupby(["speakerId", "sessionId"]).size()
# speakerId  sessionId
# 002        001           93
           # 003           93
           # 101          398
# 004        001           93
# 006        001           93
# 008        001           31
           # 002           93
           # 003           96
           # 004           31
           # 005           31
           # 006           31
           # 007           31
           # 008           31


dev["mode"].value_counts()
#audible      209   3.9*lowest
#silent        53
#whispered     53
dev["gender"].value_counts()
#male      288      10.6*lowest
#female     27
dev["transcript"].value_counts().value_counts()
# 1     79
# 3      1
# 4      7
# 5      1
# 6      8
# 16     1
# 17     8
dev["speakerId"].value_counts()
#002    156     5.7*lowest
#004     27
#006     27
#008    105     3.8*lowest
dev.groupby(["speakerId", "sessionId"]).size()
# speakerId  sessionId
# 002        001           27
           # 003           27
           # 101          102
# 004        001           27
# 006        001           27
# 008        001            9
           # 002           27
           # 003           24
           # 004            9
           # 005            9
           # 006            9
           # 007            9
           # 008            9

test["mode"].value_counts()
#audible      140   2.3*lowest
#silent        60
#whispered     60
test["gender"].value_counts()
#male      230  7.6*lowest
#female     30
test["transcript"].value_counts().value_counts()
# 26    10
test["speakerId"].value_counts()
#002     80     2.7*lowest
#004     30
#006     30
#008    120     4*lowest
test.groupby(["speakerId", "sessionId"]).size()
# speakerId  sessionId
# 002        001          30
           # 003          30
           # 101          20
# 004        001          30
# 006        001          30
# 008        001          10
           # 002          30
           # 003          30
           # 004          10
           # 005          10
           # 006          10
           # 007          10
           # 008          10

           
           

