import re
import scripts as sc
import numpy as np
import random
from collections import defaultdict

import pickle
import time
from model import *

from keras.preprocessing import sequence as ksq
from keras.layers import *
from utils import *

from json_formatter import JSONFormatter

################# configuration
np.random.seed(1)
##################

from gensim.models import word2vec
word2vec_model = word2vec.Word2Vec.load('test3100features_0minwords_10context')

def ll():
	return defaultdict(list)
def ls():
	return defaultdict(set)

class Predictor(Tracker):
	def load_weight(self, weight_file):
		assert self.exist_model
		self.model.load_weights(weight_file)

	def predict(self, pred_X, pred_X_ins):
		assert self.exist_model

		pred_info_loc = np.array([self.info_loc for i in range(pred_X.shape[0])])
		return self.model.predict([pred_X, pred_X_ins, pred_info_loc])

	def calculate_entropy(self, prob):
		print prob.shape
		entropy = np.zeros((prob.shape[0], self.slot_num))
		for i in range(prob.shape[0]):
			for j in range(prob.shape[2]):
				log_prob = np.log(prob[i, :, j])
				entropy[i][j] = -np.dot(prob[i, :, j], log_prob)

		return entropy

	# fine nearest value and corresponding cosine similarity for given word vector
	def find_nearest_value_cossim(self, vector, topic, slot):
		candidate_vectors = self.slot_value_vector[topic][slot]

		candidate_vectors_norm = np.linalg.norm(candidate_vectors, axis=1)
		candidate_vectors_norm[candidate_vectors_norm == 0] = 1
		candidate_vectors_norm = np.divide(candidate_vectors.T, candidate_vectors_norm).T

		vector_norm = np.linalg.norm(vector)
		vector_norm = vector if (vector_norm == 0) else np.divide(vector, vector_norm)

		cos_distance = np.dot(candidate_vectors_norm, vector_norm)
		nearest_loc = np.argmax(cos_distance)

		max_cos_distance = cos_distance[nearest_loc]
		max_cos_value = self.slot_value[topic][slot][nearest_loc]

		return (max_cos_value, max_cos_distance)

	# calculate total value and cosine similarity for predicted embed output
	def calculate_value_cossim(self, out_embed, dev_X_topic):
		total_value = []
		total_cossim = []

		for i in range(len(out_embed)):
			cur_value = {}

			topic = dev_X_topic[i]

			for (t, s) in self.slots:
				if t == topic:
					slot_loc = self.slot_loc[t][s]
					(nearest_value, nearest_cossim) = self.find_nearest_value_cossim(out_embed[i][slot_loc], t, s)
					cur_value[s] = nearest_value

					if s == 'INFO':
						total_cossim.append(nearest_cossim)
			total_value.append(cur_value)

		return (total_value, total_cossim)

	# calculate accuracy for given slot
	def calculate_slot_accuracy(self, slot, thres, total_value, total_entropy, total_cossim, dev_X_topic,
								dev_Y_text):
		total_segs = 0.0
		correct_segs = 0.0

		for i in range(len(dev_Y_text)):
			topic = dev_X_topic[i]
			if slot in self.slot_loc[topic].keys():
				total_segs += 1
				slot_loc = self.slot_loc[topic][slot]
				value_exist = total_cossim[i] > thres if (slot == 'INFO') else total_entropy[i][slot_loc] < thres

				if value_exist:
					if slot in dev_Y_text[i].keys():
						if total_value[i][slot] == dev_Y_text[i][slot][0]:
							correct_segs += 1
				else:
					if slot not in dev_Y_text[i].keys():
						correct_segs += 1
		accuracy = correct_segs / total_segs
		return accuracy

	# calculate fscore for given slot
	def calculate_slot_fscore(self, slot, thres, total_value, total_entropy, total_cossim, dev_X_topic, dev_Y_text):
		pred_slots = 0.0
		ref_slots = 0.0
		correct_slots = 0.0

		for i in range(len(dev_Y_text)):
			topic = dev_X_topic[i]
			if slot in self.slot_loc[topic].keys():
				slot_loc = self.slot_loc[topic][slot]
				value_exist = total_cossim[i] > thres if (slot == 'INFO') else total_entropy[i][slot_loc] < thres

				if value_exist:
					pred_slots += 1
					if slot in dev_Y_text[i].keys():
						if total_value[i][slot] == dev_Y_text[i][slot][0]:
							correct_slots += 1

				if slot in dev_Y_text[i].keys():
					ref_slots += 1

		if pred_slots == 0 or ref_slots == 0:
			return 0.
		precision = correct_slots / pred_slots
		recall = correct_slots / ref_slots

		if precision + recall == 0:
			return 0.

		fscore = 2*precision*recall / (precision+recall)
		return fscore

	# decide theshold by given total value, entropy, and cossim
	def decide_threshold(self, total_value, total_entropy, total_cossim, dev_X_topic, dev_Y_text, criteria):
		max_thres = {}
		candidate_thres = {}

		assert criteria in ['accuracy', 'fscore']

		if criteria == 'accuracy':
			criteria_function = self.calculate_slot_accuracy
		elif criteria == 'fscore':
			criteria_function = self.calculate_slot_fscore

		for topic, slot in self.slots:
			max_thres[slot] = 0.0
			candidate_thres[slot] = set()
			if slot == 'INFO':
				candidate_thres[slot].update(total_cossim)
			else:
				candidate_thres[slot].update(total_entropy[:, self.slot_loc[topic][slot]])

		slot_list = max_thres.keys()
		for slot in slot_list:
			slot_max_thres = max_thres[slot]
			slot_max_acc = criteria_function(slot, slot_max_thres, total_value, total_entropy,
															total_cossim, dev_X_topic, dev_Y_text)

			for candidate in candidate_thres[slot]:
				cur_slot_max_acc = criteria_function(slot, candidate, total_value, total_entropy, total_cossim, dev_X_topic, dev_Y_text)
				if cur_slot_max_acc > slot_max_acc:
					slot_max_thres = candidate
					slot_max_acc = cur_slot_max_acc
				max_thres[slot] = slot_max_thres
			print slot, slot_max_acc, slot_max_thres

		return max_thres

	# return total solution by decided threshold
	def get_solution(self, total_value, thresholds, total_entropy, total_cossim, dev_X_topic):
		total_solution = []
		for i in range(len(total_value)):
			topic = dev_X_topic[i]
			cur_solution = {}

			for slot in self.slot_loc[topic].keys():
				slot_loc = self.slot_loc[topic][slot]
				value_exist = total_cossim[i] > thresholds[slot] if (slot == 'INFO') else total_entropy[i][slot_loc] < \
																						  thresholds[slot]
				if value_exist:
					cur_solution[slot] = [total_value[i][slot]]

			total_solution.append(cur_solution)
		return total_solution

	# calculate final accuracy
	def calculate_accuracy(self, total_solution, dev_Y_text):
		total_segs = 0.
		correct_segs = 0.

		for i in range(len(dev_Y_text)):
			total_segs += 1
			if total_solution[i] == dev_Y_text[i]:
				correct_segs += 1
		print 'correct', correct_segs
		print 'total', total_segs
		return correct_segs / total_segs

	# print result
	def print_result(self, total_solution, SUID, X_text, X_topic, Y_text):
		correct = 0
		total = 0
		for i in range(len(SUID)):
			total += 1
			print SUID[i]
			print X_topic[i]
			print X_text[i]

			print 'My solution: ', total_solution[i]
			print Y_text[i]
			if total_solution[i] == Y_text[i]:
				print 'CORRECT'
				correct += 1
			else:
				print 'WRONG'
			print
		print correct, '/', total

	def predict_write_json(self, dstctype, datatype, thresholds, json_file):
		assert self.exist_model
		(SUID, X, X_ins, X_text, X_topic, _, Y_text) = self.load_data(dstctype, datatype, uttr_accumulate=True)
		pred = self.predict(X, X_ins)
		total_entropy = self.calculate_entropy(pred[1])
		(total_value, total_cossim) = self.calculate_value_cossim(pred[0], X_topic)
		solution = self.get_solution(total_value, thresholds, total_entropy, total_cossim, X_topic)

		#self.print_result(solution, SUID, X_text, X_topic, Y_text)
		#print self.calculate_accuracy(solution, Y_text)

		dataset_name = 'dstc' + str(dstctype) + '_' + datatype
		jf = JSONFormatter(dataset_name, 0.1, SUID, solution)
		jf.dump_to_file(json_file)
		print 'Dumped to ', json_file

import argparse, sys
def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--lstm', dest='lstm_units', action='store', type=int, default=100, help='Number of LSTM units')
	parser.add_argument('-lr', '--lr', dest='learning_rate', type=float, action='store', default=0.005, help='Learning rate')
	parser.add_argument('-dr1', '--dropout1', dest='dropout1', action='store', type=float, default=0., help='Dropout1')
	parser.add_argument('-dr2', '--dropout2', dest='dropout2', action='store', type=float, default=0., help='Dropout2')
	parser.add_argument('-e', '--epoch', dest='nb_epoch', type=int, action='store', default=300, help='Training epoch')
	parser.add_argument('-t', '--type', dest='dstctype', type=int, action='store', default=5, help='DSTC type')
	parser.add_argument('-th', '--thres', dest='decide_threshold', action='store_true', help='Check where deciding threshold or not')
	parser.add_argument('-c', '--criteria', dest='criteria', action='store', default='accuracy',
						help='Decide criteria')

	args = parser.parse_args()

	print 'LSTM units: ', args.lstm_units
	print 'Learning rate: ', args.learning_rate
	print 'Dropout: ', args.dropout1, args.dropout2
	print '--------------------------------'

	weight_file = 'dstc'+str(args.dstctype)\
					+ '_lstm'+str(args.lstm_units)\
					+ '_lr'+str(args.learning_rate)[2:]\
					+ '_dr'+str(args.dropout1)[2:]\
					+ '_'+str(args.dropout2)[2:]+'.h5'

	thres_file = 'dstc'+str(args.dstctype)\
					+ '_lstm'+str(args.lstm_units)\
					+ '_lr'+str(args.learning_rate)[2:]\
					+ '_dr'+str(args.dropout1)[2:]\
					+ '_'+str(args.dropout2)[2:]\
					+ '_'+args.criteria+'.pkl'

	dev_json_file = 'dev_dstc'+str(args.dstctype)\
					+ '_lstm'+str(args.lstm_units)\
					+ '_lr'+str(args.learning_rate)[2:]\
					+ '_dr'+str(args.dropout1)[2:]\
					+ '_'+str(args.dropout2)[2:] \
					+ '_' + args.criteria+'.json'

	test_json_file = 'test_dstc' + str(args.dstctype) \
					+ '_lstm' + str(args.lstm_units) \
					+ '_lr' + str(args.learning_rate)[2:] \
					+ '_dr' + str(args.dropout1)[2:] \
					+ '_' + str(args.dropout2)[2:] \
					+ '_' + args.criteria+ '.json'

	print 'Weight file : '+ weight_file
	print 'Threshold file : '+ thres_file


	predictor = Predictor(args.dstctype, embed_dim=100, segment_length=240)
	predictor.create_model(args.learning_rate, args.lstm_units, args.dropout1, args.dropout2)
	predictor.load_weight(weight_file)

	if args.decide_threshold:
		(_, X, X_ins, _, X_topic, _, Y_text) = predictor.load_data(args.dstctype,'dev', uttr_accumulate=False)
		pred = predictor.predict(X, X_ins)
		total_entropy = predictor.calculate_entropy(pred[1])
		(total_value, total_cossim) = predictor.calculate_value_cossim(pred[0], X_topic)
		thresholds = predictor.decide_threshold(total_value, total_entropy, total_cossim, X_topic, Y_text, args.criteria)

		pkl_file = open(thres_file, 'wb')
		pickle.dump(thresholds, pkl_file)
		pkl_file.close()
		print 'Threshold dumped to '+thres_file

	else:
		pkl_file = open(thres_file, 'rb')
		thresholds = pickle.load(pkl_file)
		pkl_file.close()

	print 'Threshold: ', thresholds
	predictor.predict_write_json(args.dstctype, 'dev', thresholds, dev_json_file)
	predictor.predict_write_json(args.dstctype, 'test', thresholds, test_json_file)


if __name__=='__main__':
	main(sys.argv)