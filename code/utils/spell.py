"""
Code adapted from https://github.com/mozilla/DeepSpeech
"""

from __future__ import absolute_import
import re
import kenlm
from heapq import heapify

# Define beam with for alt sentence search
BEAM_WIDTH = 1024
MODEL = None

# Lazy-load language model (TED corpus, Kneser-Ney, 4-gram, 30k word LM)
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = kenlm.Model('data/lm/lm.binary')
    return MODEL

def words(text):
    "List of words in text."
    return text.split()#re.findall(r'\w+', text.lower())

# Load known word set
with open('data/lm/words.txt') as f:
    WORDS = set(words(f.read()))

def log_probability(sentence):
    "Log base 10 probability of `sentence`, a list of words"
    return get_model().score(' '.join(sentence), bos = False, eos = False)

def correction(sentence):
    "Most probable spelling correction for sentence."
    layer = [(0,[])]
    for word in words(sentence):
        layer = [(-log_probability(node + next_word), node + next_word) for next_word in candidate_words(word) for priority, node in layer]
        heapify(layer)
        layer = layer[:BEAM_WIDTH]
    return ' '.join(layer[0][1])

def candidate_words(word):
    "Generate possible spelling corrections for word."
    edit_set = (known_words([word]) or known_words(edits1(word)) or known_words(edits2(word)) or split_edits(word) or [word])
    edit_list = [w.split() for w in edit_set]
    return edit_list

def known_words(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def split_edits(word):
    candidates = [word[:i] + " " + word[i:] for i in range(len(word) + 1) if word[:i] in WORDS and word[i:] in WORDS]
    return candidates

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))