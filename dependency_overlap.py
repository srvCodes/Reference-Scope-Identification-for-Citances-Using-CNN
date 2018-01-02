import re
import networkx as nx
from practnlptools.tools import Annotator

def get_shortest_path(a, b):
	text = a + b 

	annotator = Annotator()
	dep_parse = annotator.getAnnotations(text, dep_parse=True)['dep_parse']

	dp_list = dep_parse.split('\n')
	pattern = re.compile(r'.+?\((.+?), (.+?)\)')
	edges = []
	
	for dep in dp_list:
		m = pattern.search(dep)
		edges.append((m.group(1), m.group(2)))
	
	graph = nx.Graph(edges)  
	
	shortest_paths = [] 
	
	a = a.strip()
	b = b.strip()
	
	a = a.split()
	b = b.split()
	
	for i in a: 
		for j in b: 
			shortest_paths.append(nx.shortest_path_length(graph, source=i, target=j))
	
	print(shortest_paths)

a = "named entity recognizer (NER) useful many NLP applications information extraction, question answering, etc. own, NER also provide users looking person organization names quick information."
b = "automatically derived based correlation metric value used (Chieu Ng, 2002a)."

get_shortest_path(a,b)
