#Karol Wadolowski, Nikola Janjusevic
#ECE-471 Final Project

"""
This file contains a convolutional neural network that will take input images and learn
and extract features.
"""

import numpy as np 
import tensorflow as tf 

#Names of files containing training, validation, and testing data
sz = "40.npy"		#Image size
name_trainCurves 	= "Train_RegularPolyCurves" + sz
name_trainLabels 	= "Train_RegularPolyLabels" + sz 
name_validCurves 	= "Valid_RegularPolyCurves" + sz 
name_validLabels 	= "Valid_RegularPolyLabels" + sz 
name_testCurves 	= "Test_RegularPolyCurves" 	+ sz 
name_testLabels 	= "Test_RegularPolyLabels" 	+ sz 
checkpoint_dir		= "./checkpoints/"		#Save checkpoints here

class Data(object):			#Data necessary for experimentation
	def __init__(self):
		#Load data
		print("Start Loading Data")
		print("Loading Training Data")
		self.trainCurves	= np.load(name_trainCurves)
		self.trainLabels 	= np.load(name_trainLabels)
		print("Training Data Loaded")
		print("Start Loading Validation Data")
		self.validCurves 	= np.load(name_validCurves)
		self.validLabels 	= np.load(name_validLabels)
		print("Validation Data Loaded")
		print("Start Loading Test Data")
		self.testCurves 	= np.load(name_testCurves)
		self.testLabels 	= np.load(name_testLabels)
		print("Test Data Loaded")

		#Get Sizes
		self.amtTrain	= np.shape(self.trainCurves)[0]
		self.amtValid 	= np.shape(self.validCurves)[0]
		self.amtTest 	= np.shape(self.testCurves)[0]
		self.classes 	= np.shape(self.trainLabels)[1]
		
	def permuteTrain(self):
		self.index = np.arange(self.amtTrain)
		perm = np.random.choice(self.index, size = self.amtTrain)
		self.trainCurves 	= [self.trainCurves[i] for i in perm]
		self.trainLabels 	= self.trainLabels[perm,:]

	def permuteValid(self):
		self.index = np.arange(self.amtValid)
		perm = np.random.choice(self.index, size = self.amtValid)
		self.validCurves 	= [self.validCurves[i] for i in perm]
		self.validLabels 	= self.validLabels[perm,:]

data = Data()

#Data Info
trSamp 	 = data.amtTrain	#Number of training samples	
valSamp	 = data.amtValid	#Number of validation samples
testSamp = data.amtTest		#Number of test samples

#Convolutional Neural Net Structure
CL_1 	= 8					#Size of Convolutional Layer 1, layer 1
CL_2 	= 8					#Size of Convolutional Layer 2, layer 2
FC_1 	= 8 				#Size of Fully Conected Layer,	layer 3
CLASSES = data.classes 		#Size of Output Layer
WS_1 = 5					#Window Size for CL_1
WS_2 = 5					#Window Size for CL_2
STRIDE = 1

#Training and Hyperparameters
DROP 	= 0.8				#Dropout Keep Probability
LR 		= 1e-3				#Learning Rate
EPOCHS 	= 5					#Go through all the test data this many times
NUM_BATCHES = 1000			#Number of batches per epoch
BATCH_SIZE  = trSamp/NUM_BATCHES	#Number of samples per batch

print("\nStructure: ", CL_1, CL_2, FC_1, WS_1, WS_2)
print("Paramters: ", DROP, LR, EPOCHS, NUM_BATCHES, "\n")

def conv_layer(x,w,b):		#Single convolutional layer
	layer = tf.nn.conv2d(x,w,strides = [1, STRIDE, STRIDE, 1],\
		padding = 'SAME')
	layer = tf.add(layer,b)
	return tf.nn.relu(layer)

def maxPool(x):				#Pools input x
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1],\
		padding = 'SAME')

def convNet(x, dropout):	#Puts input x through the convolutional network
	#Convolutional layer
	layer_1 = conv_layer(x,weights['wCL_1'], biases['bCL_1'])
	layer_1 = maxPool(layer_1)

	#Convolutional layer
	layer_2 = conv_layer(layer_1, weights['wCL_2'], biases['bCL_2'])
	layer_2 = maxPool(layer_2)


	#Fully Connected Layer
	layer_3 = tf.reduce_mean(layer_2, axis=[1,2], keepdims=False)
	layer_3 = tf.matmul(layer_3,weights['wFC_1'])
	layer_3 = tf.add(layer_3, biases['bFC_1'])
	layer_3 = tf.nn.relu(layer_3)
	layer_3 = tf.nn.dropout(layer_3, dropout)

	return tf.add(tf.matmul(layer_3, weights['wOL']), biases['bOL'])

def slowIn(ims,labs,amt):		#Slowly Feed in the validation or test data
	lossAv = 0		#Average Batch Loss
	accAv = 0		#Average Batch Accuracy
	count = 0		#Count how mant times through the loop
	bs = amt/100	#Batch size
	for i in range(int(bs)):
		ran = np.int32(np.arange(i*bs, (i+1)*bs))
		loss_val, acc_val = sess.run([loss,acc],\
			feed_dict = {imgs: ims[ran], labels: labs[ran,:], dropout: 1})
		lossAv += loss_val
		accAv += acc_val
		count += 1
	lossAv /= count
	accAv /= count
	return [lossAv, accAv]


weights = {
	'wCL_1': tf.Variable(tf.random_normal([1, WS_1, 2, CL_1], stddev=0.1)),
	'wCL_2': tf.Variable(tf.random_normal([WS_2, WS_2, CL_1, CL_2], stddev=0.1)),
	'wFC_1': tf.Variable(tf.random_normal([CL_2, FC_1], stddev=0.1)),
	'wOL':   tf.Variable(tf.random_normal([FC_1,CLASSES]))
}

biases = {
	'bCL_1': tf.Variable(tf.zeros([CL_1])),
	'bCL_2': tf.Variable(tf.zeros([CL_2])),
	'bFC_1': tf.Variable(tf.zeros([FC_1])),
	'bOL':   tf.Variable(tf.zeros([CLASSES]))
}

imgs  	= tf.placeholder(tf.float32, [None, None, 2, 2])
labels 	= tf.placeholder(tf.float32, [None, CLASSES])
dropout = tf.placeholder(tf.float32)
label_est = convNet(imgs, dropout)

pred = tf.nn.softmax(label_est)		#Label Probabilities (prediction)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = label_est, labels = labels)
loss = tf.reduce_mean(loss)

optim = tf.train.AdamOptimizer(learning_rate = LR).minimize(loss)

#Check if prediction is correct and calculate accuracy
corr_pred = tf.equal(tf.argmax(pred, axis = 1), tf.argmax(labels, axis = 1))
acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

savePt = 0		#Save point
for epoch in range(EPOCHS):
	data.permuteTrain()
	imgs_all = data.trainCurves
	labels_all = data.trainLabels

	for batch_num in range(NUM_BATCHES):
		ran = np.int32(np.arange(batch_num*BATCH_SIZE, (batch_num+1)*BATCH_SIZE))

		#Current batch
		imgs_curr = [imgs_all[i] for i in ran]
		print(type(imgs_curr),np.shape(imgs_curr),np.shape(imgs_curr[0]))
		labels_curr = labels_all[ran,:]

		#Train on batch
		################################################################
		#Problem here turning (batch_size, m, 2, 2) into a usable value
		loss_tr, _ = sess.run([loss,optim],\
			feed_dict = {imgs:imgs_curr, labels: labels_curr, dropout: DROP})
		#################################################################

		#Check validation data and save parameters
		if (batch_num + 1) % 500 == 0:
			saver.save(sess, checkpoint_dir + 'model_' + str(savePt) + '.ckpt')
			savePt += 1

			data.permuteValid()
			imgs_val = data.validCurves
			labels_val = data.validLabels

			loss_val, acc_val = slowIn(imgs_val, labels_val, valSamp)

			print("Epoch " + str(epoch + 1) + " Batch " + str(batch_num + 1) +\
				", Training: Loss = " + "{0:.4f}".format(loss_tr) + ", Validation: Loss = " +\
				"{0:.4f}".format(loss_val) + ", Accuracy = " + "{0:.4f}".format(acc_val))
	print('')

print("Training Done")

if acc_val > .85:
	imgs_test = data.testCurves
	_, acc_test = slowIn(imgs_test, data.testLabels, testSamp)

	print("Test Accuracy = " + "{0:.4f}".format(acc_test))