# Reference-Scope-Identification-for-Citances-Using-CNN

This repository contains all files necessary to reproduce the results of my paper entitled Reference Scope Identification for Citances Using Convolutional Neural Network<sup>1</sup>.

## Feature Extraction Modules:

1. Lexical Features: *All the similarity measures require a pair of texts as input and work by averaging over all the words of the sentence.* 
- Word Overlap: 
  - [dice_coeff.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/dice_coeff.py) measures the Dice Similarity.  
  - [cosine.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/cosine.py) measures the Cosine similarity.
  - [fuzz_string_matching.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/fuzz_string_matching.py) measures Levenshtein distance based fuzzy string similarity.
  - [sequence_matcher.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/sequence_matcher.py) gives a measure of modified Gestalt pattern-matching based sequence matcher score.

2. TF-IDF similarity: [tfidf_using_cosine.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/tfidf_using_cosine.py) measures the TF-IDF vector cosine similarity between the citance and the reference sentence.

3. 
