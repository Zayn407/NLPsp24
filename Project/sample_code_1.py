#a.num_sentences function calculates scoring criterion a (number of sentences and lengths)
#b.spelling_mistakes function calculates scoring criterion b (spelling mistakes)
import pandas as pd
import numpy as np
import nltk
import spacy
from spellchecker import SpellChecker
from spacy.symbols import nsubj, VERB
from collections import Counter

# (a) Length of the essay
def score_length(essay):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay)
    num_sentences = len(list(doc.sents))
    average_length_high = 20  # Assuming that the average number of sentences in a high-scoring essay
    average_length_low = 10   # Assuming that the average number of sentences in low-scoring essays

    # Scores are assigned based on the position of the number of sentences relative to the high-low average
    if num_sentences < average_length_low:
        return 1
    elif num_sentences < average_length_high:
        return (num_sentences - average_length_low) / (average_length_high - average_length_low) * 4 + 1
    else:
        return 5
    
# (b) Spelling mistakes
def score_spelling(essay):
    spell = SpellChecker()
    # Splitting words and checking spelling
    misspelled = spell.unknown(essay.split())
    num_errors = len(misspelled)
    # Scores are determined by the number of spelling errors; the more errors the higher the score
    score = min(num_errors*0.1, 4)
    return round(score, 2)