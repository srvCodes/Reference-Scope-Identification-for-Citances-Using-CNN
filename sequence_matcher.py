# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:59:34 2017

@author: lenovo laptop
"""

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    print (similar("In such cases, neither global features (Chieu and Ng, 2002) nor aggregated contexts (Chieu and Ng, 2003) can help.", "Named Entity Recognition: A Maximum Entropy Approach Using Global Information"))
    
if __name__ == "__main__":
    main()