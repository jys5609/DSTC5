import scripts as sc
import numpy as np
import random
from collections import defaultdict
from model import *
from utils import *

################# configuration
np.random.seed(1)
##################

from gensim.models import word2vec
word2vec_model = word2vec.Word2Vec.load('test3100features_0minwords_10context')

def ll():
	return defaultdict(list)
def ls():
	return defaultdict(set)


import argparse, sys
def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--lstm', dest='lstm_units', action='store', type=int, default=100, help='Number of LSTM units')
	parser.add_argument('-lr', '--lr', dest='learning_rate', type=float, action='store', default=0.005, help='Learning rate')
	parser.add_argument('-dr1', '--dropout1', dest='dropout1', action='store', type=float, default=0., help='Dropout1')
	parser.add_argument('-dr2', '--dropout2', dest='dropout2', action='store', type=float, default=0., help='Dropout2')
	parser.add_argument('-e', '--epoch', dest='nb_epoch', type=int, action='store', default=300, help='Training epoch')
	parser.add_argument('-t', '--type', dest='dstctype', type=int, action='store', default=5, help='DSTC type')

	args = parser.parse_args()

	print 'LSTM units: ', args.lstm_units
	print 'Learning rate: ', args.learning_rate
	print 'Dropout: ', args.dropout1, args.dropout2
	print '--------------------------------'


	model = Tracker(args.dstctype, embed_dim=100, segment_length=240)
	model.create_model(args.learning_rate, args.lstm_units, args.dropout1, args.dropout2)

	(train_SUID, train_X, train_X_ins, train_X_text, train_X_topic, train_Y, train_Y_text) = model.load_data(args.dstctype, 'train', uttr_accumulate=False)
	(dev_SUID, dev_X, dev_X_ins, dev_X_text, dev_X_topic, dev_Y, dev_Y_text) = model.load_data(args.dstctype, 'dev', uttr_accumulate=False)

	weight_file = 'dstc'+str(args.dstctype)\
				  + '_lstm'+str(args.lstm_units)\
				  + '_lr'+str(args.learning_rate)[2:]\
				  + '_dr'+str(args.dropout1)[2:]\
				  + '_'+str(args.dropout2)[2:]\
				  +'.h5'

	print 'Weight file : '+ weight_file

	model.train(train_X, train_X_ins, train_Y, dev_X, dev_X_ins, dev_Y, args.nb_epoch, weight_file)

if __name__=='__main__':
	main(sys.argv)

