import numpy as np
import pandas as pd
import os
import string
import re 
import igraph
import itertools
import operator
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer # tfidf
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize # normalisation
from prettytable import PrettyTable
from collections import Counter # Counter
import argparse

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag

from library import clean_text_simple

from sklearn.feature_extraction.text import CountVectorizer
import pickle





#test = pd.read_csv(PATH_TO_DATA + 'test_set.csv', sep=',', header=0)
#test_info = pd.read_csv(PATH_TO_DATA + 'test_info.csv', sep=',', header=0)

def Read_Train_Data( first_time_run, validation_ratio = 0.2, set_nb_validation = 0):
########################################
# load files #
# The data is stored in a dictionnary  : 

# sender_mailID = {'sender@enpc.fr' : 'Email_ID'}
# mailID_content = {'Email_ID' : 'Email_content'}
# mailID_recipient = {'Email_ID : 'Recipient'}  
# mailID_date = {'Email_ID : 'Date'} 
# mailID_sender = {'Email_ID : 'Sender'}                           
########################################

	sender_mailID = {}
	mailID_content = {}
	mailID_sender = {}
	mailID_recipient = {}
	mailID_date = {}

	train_mailID = []
	valid_mailID = []

	info = PrettyTable()
	info.field_names = ['Category', 'Information']

	print 'Loading data...'
	training = pd.read_csv(PATH_TO_DATA + 'training_set.csv', sep=',', header=0)
	training_info = pd.read_csv(PATH_TO_DATA + 'training_info.csv', sep=',', header=0)

	if set_nb_validation > 4 :
		raise RuntimeError('Value Error set_nb_validation should be <= 4')

	if first_time_run : 
		path = ['sender_mailID.pkl', 'mailID_content.pkl', 'mailID_sender.pkl', 'mailID_recipient.pkl', 'mailID_date.pkl']
		path = [os.path.abspath(p) for p in path]
		print 'First time run the script, process the document, and store the information in : \n \t ----%s\n \t ----%s\n \t ----%s\n \t ----%s\n \t ----%s'%(path[0], path[1], path[2], path[3], path[4])
		print 'Put data into the dictionnaries...'
		f_sender_mailID = open('sender_mailID.pkl', 'wb')
		f_mailID_content = open('mailID_content.pkl', 'wb')
		f_mailID_sender = open('mailID_sender.pkl', 'wb')
		f_mailID_recipient = open('mailID_recipient.pkl', 'wb') 
		f_mailID_date = open('mailID_date.pkl', 'wb')

	else : 
		print 'Not the first running, only get the train email ID and valid email ID'

	for index, series in training.iterrows():
		row = series.tolist()
		sender = row[0]
		ids = map(int, row[1:][0].split(' '))
		if first_time_run : 
			sender_mailID[sender] = ids
			for emailID in ids : 
				mailID_content[emailID] = training_info[training_info['mid']==int(emailID)]['body'].tolist()[0]
				mailID_sender[emailID] = sender 
				recipients = training_info[training_info['mid']==int(emailID)]['recipients'].tolist()[0].split(' ')
				mailID_recipient[emailID] = [rec for rec in recipients if '@' in rec]
				mailID_date[emailID] = training_info[training_info['mid']==int(emailID)]['date'].tolist()[0]

		nb_mail_sender = len(ids)
		nb_valid = int(nb_mail_sender * validation_ratio)
		train_mailID = train_mailID + ids[:(set_nb_validation * nb_valid)] + ids[(set_nb_validation + 1) * nb_valid :]
		valid_mailID = valid_mailID + ids[(set_nb_validation * nb_valid) : (set_nb_validation + 1) * nb_valid]
		if index%20 == 0 :
			print 'Accomplish %d...'%index
	if first_time_run : 
		info.add_row(['Total mails nb: ', len(mailID_content)])
		info.add_row(['Total sender : ', len(sender_mailID)])
		pickle.dump(sender_mailID, f_sender_mailID)
		pickle.dump(mailID_content, f_mailID_content)
		pickle.dump(mailID_sender, f_mailID_sender)
		pickle.dump(mailID_recipient, f_mailID_recipient)
		pickle.dump(mailID_date, f_mailID_date)
		f_sender_mailID.close()
		f_mailID_content.close()
		f_mailID_sender.close()
		f_mailID_recipient.close() 
		f_mailID_date.close()
		
		
	info.add_row(['Train mails nb: ', len(train_mailID)])
	info.add_row(['Validation mails nb: ', len(valid_mailID)])
	info.add_row(['Validation ratio: ', float(len(valid_mailID))/( len(train_mailID) + len(valid_mailID))])
	print info
	

	return sender_mailID, mailID_content, mailID_sender, mailID_recipient, mailID_date, train_mailID, valid_mailID

## Preprocessing the text
## remove punctuation, numbers, stop words etc. 
def Preprocessing(mailID_content, preprocessing_file) :
	if os.path.exists(os.path.abspath(preprocessing_file)) : 
		print 'The text is already be processed in %s' %os.path.abspath(preprocessing_file)
		f = open(os.path.abspath(preprocessing_file), 'rb')
		content = pickle.load(f)
		f.close()
		return content
	else :
		print 'You choose a new processing method for text, the text will be stored in %s'%os.path.abspath(preprocessing_file)
		keys, values = mailID_content.keys(), mailID_content.values() 
		for i,content in enumerate(values) : 
			content = clean_text_simple(content)
			content = [word for word in content if len(word) > 2]
			values[i] = ' '.join(content)
			if i%10000 ==0 : 
				print 'process %d emails already...'%i
		content = dict(zip(keys, values))
		f = open(os.path.abspath(preprocessing_file), 'wb')
		pickle.dump(content, f)
		f.close()
	return content
		


## Extract tfidf feature for train data set
def Extract_tfidf_train_feature(train_mailID, mailID_content, normalised = True) : 
	train_text = [mailID_content[emailID] for emailID in train_mailID]
	vectorizer = CountVectorizer(min_df = 1)
	texts_fit = vectorizer.fit_transform(train_text) ## convert to vector
	transformer = TfidfTransformer(smooth_idf=False)
	train_feature = transformer.fit_transform(texts_fit) 
	index_keep = np.array(train_feature.sum(0).argsort())[0, -NB_WORDS : ]
	train_feature = train_feature[:, index_keep].toarray()
	if normalised : 
		train_feature = normalize(train_feature, axis = 1)
	return train_feature, vectorizer, index_keep

## Extract tfidf feature for validation or test data set
def Extract_tfidf_valid_test_feature(vectorizer, mailID, mailID_content, index_keep, normalised = True) :
	text = [mailID_content[ids] for ids in mailID]
	text_fit = vectorizer.transform(text)
	transformer = TfidfTransformer(smooth_idf=False)
	feature = transformer.fit_transform(text_fit)
	feature = feature[:, index_keep].toarray() 
	if normalised : 
		feature = normalize(feature, axis = 1)
	return feature

## Build a graph only keep NB_CONNECTIONS edges for each node
## it is a dictionnary sender_recipient store {sender : {recipient : [emailID1, emailID2...]}}

def Build_graph(sender_mailID, mailID_recipient, train_mailID, mailID_sender) : 

	sender_recipient = {}

	for mailID in train_mailID : 
		sender = mailID_sender[mailID]
		if not sender_recipient.has_key(sender) : 
			sender_recipient[sender] ={}

		for recipient in mailID_recipient[mailID] : 
			if not sender_recipient[sender].has_key(recipient):
				sender_recipient[sender][recipient] = [mailID]
			else : 
				sender_recipient[sender][recipient].append(mailID)
	for sender in sender_recipient.keys() : 
		list_sorted = sorted(sender_recipient[sender].items(), key=lambda x : len(x[1]), reverse = True)
		if len(list_sorted) > NB_CONNECTIONS : 
			list_sorted = list_sorted[:NB_CONNECTIONS]
		sender_recipient[sender] = dict(list_sorted)

	return sender_recipient
		

def Prediction(sender_recipient, train_feature, prediction_feature, train_mailID, prediction_mailID, mailID_sender, sender_mailID, consider_recipient_as_sender = True):

	mailID_prediction = {}
	train_mailID = np.array(train_mailID)
	prediction_mailID = np.array(prediction_mailID)
	for i, mailID in enumerate(prediction_mailID) : 
		sender = mailID_sender[mailID]
		feature = prediction_feature[i]
		dict_recipient = sender_recipient[sender]
		scores = dict(zip(dict_recipient.keys(), [0] * len(dict_recipient)))
		for recipient in dict_recipient.keys() : 
			ID = dict_recipient[recipient] ## mail that recipient reveceives
			if consider_recipient_as_sender : 
				ID += sender_mailID[recipient] if sender_mailID.has_key(recipient) else []  ## mail that recipient sends
				ID = list(set(ID) & set(train_mailID))
			#ID = sender_mailID[recipient] ## mail that recipient send
			index_ID_train_feature = np.array([np.where(train_mailID == ids)[0][0] for ids in ID]) 
			scores[recipient] = train_feature[index_ID_train_feature].dot(feature.reshape((-1, 1))).max()
		
		sorted_scores = sorted(scores.items(), key=lambda x : x[1], reverse = True)
		if len(sorted_scores) > K :
			sorted_scores = sorted_scores[:K]
		mailID_prediction[mailID] = [key[0] for key in sorted_scores]
		if i % 1000 == 0:
			print 'Precition process : %d mails'%i
		
	return mailID_prediction

def Evaluation(mailID_prediction, mailID_recipient, printDetail = True) :
	score = np.zeros(len(mailID_prediction))
	## get some statistic about the result
	info = PrettyTable()
	info.field_names = ['Category', 'Information']
	total_find_nb = 0
	total_prediction = 0
	total_mail = len(mailID_prediction)
	percentage_find_position = np.zeros(K)	
	res = np.zeros( K + 5)

	for n, (ID, prediction) in enumerate(mailID_prediction.items()):
		ground_truth = mailID_recipient[ID]
		score_item = []
		find_number = 0.0
		for i,p in enumerate(prediction) : 
			if p in ground_truth :
				find_number += 1 
				score_item.append(find_number/(i+1))
				percentage_find_position[i] += 1
		total_find_nb += find_number
		total_prediction += i
		score[n] = 0 if len(score_item) < 1 else np.mean(score_item)

	res[:4] = np.array([total_mail, total_prediction, int(total_find_nb), float(total_find_nb)/total_prediction * 100])
	info.add_row(['Total email in prediction :', res[0]])
	info.add_row(['Total predictions :', res[1]])
	info.add_row(['Total recipients found :', res[2]])
	info.add_row(['Find percentage:', '%.2f %%'%(res[3]) ])
	
	res[4 : -1] = percentage_find_position[:]/float(total_find_nb) * 100
	for i,pos in enumerate(percentage_find_position) : 
		s = 'Recipients found in %d position :' %(i+1)
		percentage = '%.2f %%'%(res[4 + i])
		info.add_row([s, percentage])

	res[-1] = score.mean()
	info.add_row(['Validation @10 is ', "{0:.5f}".format(res[-1])])
	if printDetail : 
		print info
	
	return res

def Cross_Validation(first_time_run, preprocessing_file, consider_recipient_as_sender, printDetail = False, normalised = True): 

	if first_time_run :
		sender_mailID, mailID_content, mailID_sender, mailID_recipient, mailID_date, _, _ = Read_Train_Data(first_time_run)
		mailID_content = Preprocessing(mailID_content, preprocessing_file)
	else : 
		f_sender_mailID = open('sender_mailID.pkl', 'rb')
		f_mailID_content = open('mailID_content.pkl', 'rb')
		f_mailID_sender = open('mailID_sender.pkl', 'rb')
		f_mailID_recipient = open('mailID_recipient.pkl', 'rb') 
		f_mailID_date = open('mailID_date.pkl', 'rb')
		sender_mailID = pickle.load(f_sender_mailID) 
		mailID_content = pickle.load(f_mailID_content) 
		mailID_sender= pickle.load(f_mailID_sender)
		mailID_recipient = pickle.load(f_mailID_recipient)
		mailID_date = pickle.load(f_mailID_date)
		mailID_content = Preprocessing(mailID_content, preprocessing_file)
		f_sender_mailID.close()
		f_mailID_content.close()
		f_mailID_sender.close()
		f_mailID_recipient.close() 
		f_mailID_date.close()

	cv_info = np.zeros((5, K+5))
	for nb_set in range(5) :
		_, _, _, _, _, train_mailID, valid_mailID = Read_Train_Data(first_time_run = False, validation_ratio = 0.2, set_nb_validation = nb_set)
		train_feature, vectorizer, index_keep = Extract_tfidf_train_feature(train_mailID, mailID_content, normalised )
		valid_feature = Extract_tfidf_valid_test_feature(vectorizer, valid_mailID, mailID_content, index_keep, normalised)
		sender_recipient = Build_graph(sender_mailID, mailID_recipient, train_mailID, mailID_sender)
		mailID_prediction = Prediction(sender_recipient, train_feature, valid_feature, train_mailID, valid_mailID, mailID_sender, sender_mailID, consider_recipient_as_sender)
		cv_info[nb_set, :] = Evaluation(mailID_prediction, mailID_recipient, printDetail)
	

	info = PrettyTable()
	info.add_column('Category', ['Total email in prediction', 'Total predictions', 'Total recipients found', 
					'Find percentage:', 'Recipients found in 1st position', 'Recipients found in 2nd position', 
					'Recipients found in 3rd position', 'Recipients found in 4th position', 'Recipients found in 5th position',
					'Recipients found in 6th position', 'Recipients found in 7th position', 'Recipients found in 8th position',
					'Recipients found in 9th position', 'Recipients found in 10th position', 'Score @10'])
	for nb_set in range(5) : 
		info.add_column('Cross Validation %d'%(nb_set+1), cv_info[nb_set, :].tolist())

	info.add_column('Average', np.mean(cv_info, axis = 0).tolist())
	print info


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print '******--------- KAGGLE CHALLENGE : Prediction of Email recipient  ------*******'
	print '\n Data challenge project on \" Email recipient recommendation  \" '

	
	# global setup settings, and checkpoints
	parser.add_argument('-f', '--first_time_run', dest='first_time_run', type=int, help='First time run the algo??? ')
	parser.add_argument('-p', '--preprocessing_file', dest='preprocessing_file', type=str, help='file path to read/write the text after preprocessing')
	parser.add_argument('-c', '--consider_recipient_as_sender', dest='consider_recipient_as_sender', type=int, help='Consider each recipient as a message sender ???')
	parser.add_argument('-d', '--printDetail', dest='printDetail', type=int, help='Print detail on Cross Validation???')
	parser.add_argument('-n', '--normalised', dest='normalised', type=int, help='Normalization of feature ???')
	parser.add_argument('-nw', '--NB_WORDS', dest='NB_WORDS', type=int, help='Vocabulary size')
	parser.add_argument('-nc', '--NB_CONNECTIONS', dest='NB_CONNECTIONS', type=int, help='Nomber of connections kept for each node')
	
	
	
	
	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	

	NB_WORDS =  params['NB_WORDS'] ## VOCABULARY SIZE
	K = 10 ## PREDICTION SIZE
	NB_CONNECTIONS = params['NB_CONNECTIONS'] ## NB OF CONNECTIONS KEPT FOR EACH SENDER

	PATH_TO_RESULTS = '/home/xt/xi/mva/Text_Graph/project/'
	PATH_TO_DATA = '/home/xt/xi/mva/Text_Graph/project/'


	Cross_Validation(first_time_run = params['first_time_run'], 
					preprocessing_file = params['preprocessing_file'], 
					consider_recipient_as_sender = params['consider_recipient_as_sender'], 
					printDetail = params['printDetail'], 
					normalised = params['normalised'])
	
		
		
		
		
		
		
		
		
	

	

	
