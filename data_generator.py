import re
import scripts as sc
import numpy as np
import random
import nltk
from collections import defaultdict

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.stem.porter import *
stemmer = PorterStemmer()

################# configuration
np.random.seed(1)
dstctype = 5
embed_dim = 100
##################

from gensim.models import word2vec
word2vec_model = word2vec.Word2Vec.load('test3100features_0minwords_10context')

#onto_file = 'scripts/config/ontology_dstc'+str(dstctype)+'.json'
onto_file = 'scripts/config/ontology_dstc5.json' ## TODO: DSTC4 ontology
tagsets = sc.ontology_reader.OntologyReader(onto_file).get_tagsets()

def ll():
	return defaultdict(list)

def ls():
	return defaultdict(set)

def w_normalize(tokens):
	data = tokens.lower()
	data = re.sub('[^0-9a-z ]+', ' ', data)
	data = [lemmatizer.lemmatize(w) for w in data.split()]  #TODO: what about stemmer?
	return data

def get_index(w):
	if w in word2vec_model.vocab:
		return word2vec_model[w]
	else:
		return np.zeros(embed_dim)

def get_indices(x):
	return np.array([get_index(w) for w in x])

def get_vector(x):
	return np.sum(get_indices(x), axis=0)


print 'building vocabulary'

slots = [] # [[TOPIC, SLOT]]
slot_loc = {} # {TOPIC: {SLOT: slot location number}}
i = 0
for topic in tagsets.keys():
	slot_loc[topic] = {}
	for slot in tagsets[topic].keys():
		slots.append([topic, slot])
		slot_loc[topic][slot] = i
		i+=1

slot_num = len(slots)

slot_value = defaultdict(ll)	# {TOPIC: {SLOT: [slot value list]}}
slot_value_vector = defaultdict(ll) # {TOPIC: {SLOT: [slot vector list]}}
slot_voc = defaultdict(ls) # {TOPIC: {SLOT: {w_normalized slot value set}}}

for topic in tagsets:
	for slot in tagsets[topic]:
		for value in tagsets[topic][slot]:
			wnormed = w_normalize(value)
			vec = get_vector(wnormed)

			slot_value[topic][slot].append(value)
			slot_value_vector[topic][slot].append(vec)
			slot_voc[topic][slot].update(wnormed)

		slot_voc[topic][slot].update(w_normalize(slot))
		slot_voc[topic][slot].update(w_normalize(topic))

# transform list of 1d vectors to 2d matrix
for topic in tagsets:
	for slot in tagsets[topic]:
		slot_value_vector[topic][slot] = np.array(slot_value_vector[topic][slot])

# update total ontology vocabulary
onto_voc = set()
for topic in tagsets:
	for slot in tagsets[topic]:
		onto_voc.update(slot_voc[topic][slot])

print 'complete ontology vocabulary'

def get_lvec(label, topic):

	answer_values = [np.zeros(embed_dim) for x in range(slot_num)]

	for slot, value in label.iteritems():

		if not (slot in slot_loc[topic].keys()):
			continue
		slot_where = slot_loc[topic][slot]

		answer_values[slot_where] = get_vector(w_normalize(value[0]))
	return answer_values

def get_word_in_slot(w, topic):

	# consider if the word w exist in each slot vocabulary
	# False: -1, True: 1

	res = -np.ones(slot_num)

	for slot in slot_loc[topic].keys():
		if w in slot_voc[topic][slot]:
			res[slot_loc[topic][slot]] = 1

	return res


def get_in_slot(x, topic):
	return np.array([get_word_in_slot(w, topic) for w in x])

def data_walker(dataset, lang):
	uttr_segs = defaultdict(list)
	topic_segs = {}
	label_segs = {}

	for session in dataset:
		sid = session.log['session_id']
		segment_id = None
		for (uttr, trans, label) in session:
			uid = uttr['utter_index']
			target_bio = uttr['segment_info']['target_bio'].lower()
			if target_bio in ['b', 'o']:
				if segment_id is not None:
					print len(uttr_segs), '------', segment_id
					segment_id = None
			if target_bio == 'o':
				continue
			if segment_id is None:
				segment_id = 's'+str(sid).zfill(5)+'u'+str(uid).zfill(5)
				label_segs[segment_id] = label['frame_label']
				topic_segs[segment_id] = uttr['segment_info']['topic']
			if lang == 'en':
				#uttr_segs[segment_id].append(uttr['transcript']+['%EOU'])
				uttr_segs[segment_id].append(w_normalize(uttr['transcript'])+['%EOU'])
			else:
				if len(trans['translated'])>0:
					#uttr_segs[segment_id].append(trans['translated'][0]['hyp'])
					uttr_segs[segment_id].append(w_normalize(trans['translated'][0]['hyp'])+['%EOU'])
	return (uttr_segs, label_segs, topic_segs)

def data_generate(dataset, lang, uttr_accumulate):
	(uttr_segs, label_segs, topic_segs) = data_walker(dataset, lang)

	SUID = []
	X = [] # list of (utterance length)
	X_ins = [] # list of np.array(utterance length * # of slots)
	X_text = []
	X_topic = []
	Y = [] # list of pairs of binary slot-existence vector and (# slots, vector_size)
	Y_text = []


	for suid in sorted(uttr_segs.keys()):
		topic = topic_segs[suid]
		y_text = label_segs[suid]
		y = get_lvec(y_text, topic)

		if uttr_accumulate:
			x = []
			sid = int(suid[1:6])
			uid = int(suid[7:])
			for uttr in uttr_segs[suid]:
				id = 's' + str(sid).zfill(5) + 'u' + str(uid).zfill(5)
				SUID.append(id)
				x += uttr

				X.append(get_indices(x))
				X_ins.append(get_in_slot(x, topic))
				X_topic.append(topic)
				x_text = ' '.join(x)
				x_text = x_text.replace('%EOU', '\n')
				X_text.append(x_text)

				Y.append(y)
				Y_text.append(y_text)
				uid += 1
		else:
			x = [item for sublist in uttr_segs[suid] for item in sublist]

			SUID.append(suid)
			X.append(get_indices(x))
			X_ins.append(get_in_slot(x, topic))
			X_text.append(uttr_segs[suid])
			X_topic.append(topic)

			y = label_segs[suid]
			Y.append(get_lvec(y, topic))
			Y_text.append(y)

	return (SUID, X, X_ins, X_text, X_topic, Y, Y_text)

import pickle
def write_file(dstctype, datatype, lang, uttr_accumulate):
	if lang == 'en':
		dataset = sc.dataset_walker('dstc' + str(dstctype) + '_'+datatype, dataroot='data', translations=False, labels=True)
	else:
		dataset = sc.dataset_walker('dstc' + str(dstctype) + '_' + datatype, dataroot='data', translations=True, labels=True)

	if uttr_accumulate:
		filename = 'dstc' + str(dstctype) + '_' + datatype + '_acc.pkl'
	else:
		filename = 'dstc' + str(dstctype) + '_' + datatype + '.pkl'
	(SUID, X, X_ins, X_text, X_topic, Y, Y_text) = data_generate(dataset, lang, uttr_accumulate)

	pkl_file = open(filename, 'wb')
	pickle.dump(SUID, pkl_file)
	pickle.dump(X, pkl_file)
	pickle.dump(X_ins, pkl_file)
	pickle.dump(X_text, pkl_file)
	pickle.dump(X_topic, pkl_file)
	pickle.dump(Y, pkl_file)
	pickle.dump(Y_text, pkl_file)
	pkl_file.close()
	print 'data dumped to '+filename


pkl_general = open('dstc'+str(dstctype)+'_general.pkl', 'wb')
pickle.dump(slots, pkl_general)
pickle.dump(slot_loc, pkl_general)
pickle.dump(slot_value, pkl_general)
pickle.dump(slot_value_vector, pkl_general)


if dstctype==5:
	write_file(5, 'train','en', False)
	write_file(5, 'dev', 'cn', False)
	write_file(5, 'dev', 'cn', True)
	write_file(5, 'test', 'cn', True)
else:
	write_file(4, 'train','en', False)
	write_file(4, 'dev', 'en', False)
	write_file(4, 'dev', 'en', True)
	write_file(4, 'test','en', True)
