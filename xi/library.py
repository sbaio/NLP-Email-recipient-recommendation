import string
import re
import string
import itertools
import copy
import igraph
import operator
import nltk
from string import maketrans
from nltk.corpus import stopwords
from nltk import pos_tag


def clean_text_simple(text, remove_stopwords=True, pos_filtering=True, stemming=True):
    
    punct = string.punctuation.replace('-', '') + '\n\t'
    numbers = '1234567890'
    text = text.translate(maketrans(numbers,"          "), punct) # remove punctuation (preserving intra-word dashes)
	
    
    # convert to lower case
    text = text.lower()
    
    # strip extra white space
    text = re.sub(' +',' ',text)

    # strip leading and trailing white space
    text = text.strip()

    # tokenize (split based on whitespace)
    tokens = nltk.word_tokenize(text)
    if pos_filtering == True:
        # apply POS-tagging
        tagged_tokens = pos_tag(tokens)
        # retain only nouns and adjectives
        tokens_keep = []
        for i in range(len(tagged_tokens)):
            item = tagged_tokens[i]
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        stpwds = stopwords.words('english')
        # remove stopwords
        tokens = [word for word in tokens if word not in stpwds]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            try : 
            	tokens_stemmed.append(stemmer.stem(token))
            except:
				print 'token is %s 3 length, and ended with -ed'%token
				tokens_stemmed.append(token)
        tokens = tokens_stemmed

    return tokens

	
def terms_to_graph(terms, w):
    # This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    # Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'
    
    from_to = {}
    
    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))
    
    new_edges = []
    
    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
    
    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in xrange(w, len(terms)):
        # term to consider
        considered_term = terms[i] # the ith term
        # all terms within sliding window
        terms_temp = terms[(i-w+1):(i+1)] # the i-3 th, i-2 th, i-1 th and ith
        
        # edges to try
        candidate_edges = []
        for p in xrange(w-1):
            candidate_edges.append((terms_temp[p],considered_term))
    
        for try_edge in candidate_edges:
            
            # if not self-edge
            if try_edge[1] != try_edge[0]:
            
                # if edge has already been seen, update its weight
                if from_to.has_key(try_edge) : 
					from_to[try_edge] += 1
                                   
                # if edge has never been seen, create it and assign it a unit weight     
                else:
					from_to[try_edge] = 1
					new_edges.append(try_edge)
    
    # create empty graph
    g = igraph.Graph(directed=True)
    
    # add vertices
    g.add_vertices(sorted(set(terms)))
    
    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())
    
    # set edge and vertice weights
    g.es['weight'] = from_to.values() # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=from_to.values()) # weighted degree
    
    return g


def unweighted_k_core(g):
    # work on clone of g to preserve g 
 	gg = copy.deepcopy(g)    
    
    # initialize dictionary that will contain the core numbers
	cores_g = dict(zip(gg.vs['name'],[0]*len(gg.vs)))
    
	i = 0
    
    # while there are vertices remaining in the graph
	while len(gg.vs)>0:
        ### fill the gaps ###
		for v in gg.vs:
			if v.strength() <= i :
				cores_g[v['name']] = i
				gg.delete_vertices(v)
		i += 1
	return cores_g
	

def accuracy_metrics(candidate, truth):
    
    # true positives ('hits') are both in candidate and in truth
    tp = len(set(candidate).intersection(truth))
    
    # false positives ('false alarms') are in candidate but not in truth
    fp = len([element for element in candidate if element not in truth])
    
    # false negatives ('misses') are in truth but not in candidate
    ### fill the gap ###

    # precision
    prec = round(float(tp)/(tp+fp),5)
    
    # recall
    ### fill the gap ###
    
    if prec+rec != 0:
        # F1 score
        f1 = round(2 * float(prec*rec)/(prec+rec),5)
    else:
        f1=0
       
    return (prec, rec, f1)
