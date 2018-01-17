# Reference-Scope-Identification-for-Citances-Using-CNN

This repository contains all files necessary to reproduce the results of our paper "Reference Scope Identification for Citances Using Convolutional Neural Network<sup>1</sup>".
## Reference-Citance Pair Extraction:
[generate_test.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/generate_test.py) along with test*.py are meant for data set parsing.

**Stopwords removal**: [remove_stopwords.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/remove_stopwords.py)

## Feature Extraction Modules:

1. **Lexical Features**: *All the similarity measures require a pair of texts as input and work by averaging over all the words of the sentence.* 
- Word Overlap Measures: 
  - [dice_coeff.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/dice_coeff.py) measures the Dice Similarity.  
  - [cosine.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/cosine.py) measures the Cosine similarity.
  - [jaccard_sim.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/jaccard_sim.py) measures the Jaccard similarity.
  - [fuzz_string_matching.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/fuzz_string_matching.py) measures Levenshtein distance based fuzzy string similarity.
  - [sequence_matcher.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/sequence_matcher.py) gives a measure of modified Gestalt pattern-matching based sequence matcher score.

- TF-IDF similarity: [tfidf_using_cosine.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/tfidf_using_cosine.py) measures the TF-IDF vector cosine similarity between the citance and the reference sentence.

- ROUGE measure: [rouge_score.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/rouge_score.py) gives a measure of ROUGE-1, ROUGE-2 and ROUGE-L metrics.

- Named entity overlap: [ner_overlap.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/ner_overlap.py)


2. **Knowledge-based Feature**: [wordnet_similariy.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/wordnet_similarity.py) measures the best semantic similarity score between words in the citance and the reference sentence out of all the sets of cognitive synonyms (synsets) present in the WordNet.

3. **Corpus-based Feature**: [word2vec_similarity.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/word2vec_similarity.py), as the name denotes.

4. **Surface Features**: 
- [surface_features.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/surface_features.py)
- [sentiWordNet.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/sentiWordNet.py) measures the overall positive and negative sentiment score of the reference sentence averaged over all the words, based on the SentiWordNet 3.0 lexical resource.
- [yuleK.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/yuleK.py) measures lexical richness of the reference sentence based on Yule’s K index.

## Classification Algorithm: 
[main.py](https://github.com/Saurav0074/Reference-Scope-Identification-for-Citances-Using-CNN/blob/master/main.py) contains the 1-D CNN implementation along with GBC and ABC classifiers for training and testing the above generated feature vectors.

The original presentation can be found [here](https://www.slideshare.net/SauravJha28/reference-scope-identification-of-citances-using-convolutional-neural-network).

In case of queries, feel free to reach out at mail@sauravjha.com.np. 

# References
- S. Jha, A. Chaurasia, A. Sudhakar, and A. K. Singh, “Reference scope identification for citances using
convolutional neural networks,” in Proceedings of the 14th International Conference on Natural Language
Processing (ICON-2017). Kolkata, India: NLP Association of India, December 2017, pp. 23–32. [Online].
Available: http://www.aclweb.org/anthology/W/W17/W17-7504
