import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="SJE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-mr', '--margin', default=1, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)

"""

Best Values of (norm, lr, margin) found by validation & corr. test accuracies:

CUB  -> (std, 0.1, 4.0)  -> Test Acc : 0.4938
AWA1 -> (L2, 1.0, 2.5)   -> Test Acc : 0.5890
AWA2 -> (L2, 1.0, 2.5)   -> Test Acc : 0.5830
SUN  -> (std, 1.0, 2.0)  -> Test Acc : 0.5347
APY  -> (None, 0.01, 1.5)-> Test Acc : 0.3286

"""

class SJE():
	
	def __init__(self, args):

		self.args = args

		random.seed(self.args.rand_seed)
		np.random.seed(self.args.rand_seed)

		data_folder = '../xlsa17/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
		self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		self.labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
		self.labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
		self.labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])

		train_labels_seen = np.unique(self.labels_train)
		val_labels_unseen = np.unique(self.labels_val)
		test_labels_unseen = np.unique(self.labels_test)

		i=0
		for labels in train_labels_seen:
			self.labels_train[self.labels_train == labels] = i    
			i+=1
		
		j=0
		for labels in val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j+=1
		
		k=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = k
			k+=1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

		if self.args.norm_type=='std':
			scaler = preprocessing.StandardScaler()
			scaler.fit(self.X_train.T)

			self.X_train = scaler.transform(self.X_train.T).T
			self.X_val = scaler.transform(self.X_val.T).T
			self.X_test = scaler.transform(self.X_test.T).T

		if self.args.norm_type=='L2':
			self.X_train = self.normalizeFeature(self.X_train.T).T
			# self.X_val = self.normalizeFeature(self.X_val.T).T
			# self.X_test = self.normalizeFeature(self.X_test.T).T

	def normalizeFeature(self, x):
	    # x = N x d (d:feature dimension, N:number of instances)
	    x = x + 1e-10
	    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	    feat = x / feature_norm[:, np.newaxis]

	    return feat

	def find_compatible_y(self, X_n, W, y_n):

		XW = np.dot(X_n, W)
		# Scale the projected vector
		# XW = preprocessing.scale(XW) # TODO: is this a bug because this isn't L2 normalization?
		XW = XW / np.linalg.norm(XW)
		scores = np.zeros(self.train_sig.shape[1])
		scores[y_n] = 0.0
		gt_class_score = np.dot(XW, self.train_sig[:, y_n])
		
		for i in range(self.train_sig.shape[1]):
			if i!=y_n:
				scores[i] = self.args.margin + np.dot(XW, self.train_sig[:, i]) - gt_class_score

		return np.argmax(scores)

	def update_W(self, W, idx):
		
		for j in idx:
			X_n = self.X_train[:, j]
			y_n = self.labels_train[j]
			y = self.find_compatible_y(X_n, W, y_n)
			
			if y!=y_n:
				Y = np.expand_dims(self.train_sig[:, y_n]-self.train_sig[:, y], axis=0)
				W += self.args.lr*np.dot(np.expand_dims(X_n, axis=1), Y)
		
		return W

	def fit(self):

		print('Training...\n')

		best_val_acc = 0.0
		best_tr_acc = 0.0
		best_val_ep = -1
		best_tr_ep = -1
		
		rand_idx = np.arange(self.X_train.shape[1])

		W = np.random.rand(self.X_train.shape[0], self.train_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		for ep in range(self.args.epochs):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(W, rand_idx)
			
			val_acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
			tr_acc = self.zsl_acc(self.X_train, W, self.labels_train, self.train_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, val_acc, elapsed//60, elapsed%60))
			
			if val_acc>best_val_acc:
				best_val_acc = val_acc
				best_val_ep = ep+1
				best_W = np.copy(W)
			
			if tr_acc>best_tr_acc:
				best_tr_ep = ep+1
				best_tr_acc = tr_acc

			if ep+1-best_val_ep>self.args.early_stop:
				print('Early Stopping by {} epochs. Exiting...'.format(self.args.epochs-(ep+1)))
				break

		print('\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
		
		return best_W

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.array([np.argmax(output) for output in dist])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		return acc

	def evaluate(self):

		best_W = self.fit()

		print('Testing...\n')

		test_acc = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)

		print('Test Acc:{}'.format(test_acc))

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = SJE(args)	
	clf.evaluate()
