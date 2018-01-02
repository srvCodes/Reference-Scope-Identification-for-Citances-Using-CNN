from rouge import Rouge 

hypothesis = "In such cases, neither global features (Chieu and Ng, 2002) nor aggregated contexts (Chieu and Ng, 2003) can help."

reference = "Global features are extracted from other occurrences of the same token in the whole document."

def calc_Rouge(a,b):
	rouge = Rouge()
	scores = rouge.get_scores(a, b)

	rougelist = []

	for key, value in scores[0]['rouge-1'].items():
		rougelist.append(value)

	for key, value in scores[0]['rouge-2'].items():
		rougelist.append(value)

	for key, value in scores[0]['rouge-l'].items():
		rougelist.append(value)

	return tuple(rougelist)
