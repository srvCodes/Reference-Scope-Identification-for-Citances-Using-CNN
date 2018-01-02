from fuzzywuzzy import fuzz
from fuzzywuzzy import process

#print(fuzz.ratio("this is a test", "this is a pre-test!"))

def get_string_similarity(i, j):
	return fuzz.ratio(i, j)