#c.i agreement function calculates scoring criterion c.i (agreement within the sentence)
#c.ii verbs function calculates scoring criterion c.ii (verb mistakes)
import pandas as pd
import numpy as np
import nltk
import spacy
from spellchecker import SpellChecker
from spacy.symbols import nsubj, VERB
from collections import Counter
# (c) Syntax/Grammar
#     
#     (i) Subject-Verb agreement 
def score_subject_verb_agreement(essay):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay)
    agreement_errors = 0
    
    for token in doc:
        if token.pos == VERB:
            subjects = [child for child in token.children if child.dep == nsubj]
            for subject in subjects:
                # Singular subjects should correspond to singular verb forms
                if subject.tag_ in ['NN', 'NNP'] and token.tag_ != 'VBZ':
                    agreement_errors += 1
                # Plural subjects should correspond to plural verb forms
                elif subject.tag_ in ['NNS', 'NNPS'] and token.tag_ != 'VBP':
                    agreement_errors += 1

    # Score is calculated based on the total number of sentences in the essay and the number of errors, ranging from 1 to 5
    num_sentences = len(list(doc.sents))
    # If there are errors, each error reduces the score based on the total number of sentences, but the score remains at least 1
    score = max(5 - (agreement_errors * 5 / num_sentences if num_sentences > 0 else 0), 1)
    return round(score, 2)  # Rounded to two decimal places

#     (ii) Verb tense / missing verb / extra verb 
def score_verb_usage(essay):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay)
    verb_errors = 0
    
    for sentence in doc.sents:
        has_aux = False
        has_main_verb = False
        for token in sentence:
            # Detect auxiliary verbs
            if token.tag_ == "MD":
                has_aux = True
            # Detect main verbs
            if token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
                has_main_verb = True

            # If there is an auxiliary verb but no main verb or another auxiliary, count as an error
            if has_aux and not has_main_verb and token.pos_ != "VERB":
                verb_errors += 1

            # If the sentence ends but lacks a main verb, also count as an error
            if token == sentence[-1] and not has_main_verb:
                verb_errors += 1

            # More detection logic can be added here, such as checking for tense errors, etc.

    num_sentences = len(list(doc.sents))
    # Convert errors to score, more errors result in lower scores
    score = max(5 - (verb_errors * 2 / num_sentences if num_sentences > 0 else 0), 1)
    return round(score, 2)  # Rounded to two decimal places
