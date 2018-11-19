#Karol Wadolowski, Nikola Janjusevic
#ECE-471 Final Project

"""
This file contains a convolutional neural network that will take input images and learn
and extract features.
"""

import numpy as np 
import tensorflow as tf 

#Names of files containing training, validation, and testing data
name_trainIMGs 		= "Train_RegularPolyImgs128.npy"
name_trainLabels 	= "Train_RegularPolyLabels128.npy"
name_validIMGs 		= "Valid_RegularPolyImgs128.npy"
name_validLabels 	= "Valid_RegularPolyLabels128.npy"
name_testIMGs 		= "Test_RegularPolyImgs128.npy"
name_testLabels 	= "Test_RegularPolyLabels128.npy"
checkpoint_dir		= "./checkpoints/"		#Save checkpoints here

class Data(object):			#Data necessary for experimentation
	def __init__(self):
		#Load data
		print("Start Loading Data")
		print("Loading Training Data")
		self.trainIMGs 		= np.load(name_trainIMGs)
		self.trainLabels 	= np.load(name_trainLabels)
		print("Training Data Loaded")
		print("Start Loading Validation Data")
		self.validIMGs 		= np.load(name_validIMGs)
		self.validLabels 	= np.load(name_validLabels)
		print("Validation Data Loaded")
		print("Start Loading Test Data")
		self.testIMGs 		= np.load(name_testIMGs)
		self.testLabels 	= np.load(name_testLabels)
		print("Test Data Loaded")

		#Get Sizes
		[self.amtTrain, self.imgH, self.imgW] = np.shape(self.trainIMGs)
		self.amtValid 	= np.shape(self.validIMGs)[0]
		self.amtTest 	= np.shape(self.testIMGs)[0]
		self.classes 	= np.shape(self.trainLabels)[1]
		
	def permuteTrain(self):
		self.index = np.arange(self.amtTrain)
		perm = np.random.choice(self.index, size = self.amtTrain)
		self.trainIMGs 		= self.trainIMGs[perm,:,:]
		self.trainLabels 	= self.trainLabels[perm,:]

	def permuteValid(self):
		self.index = np.arange(self.amtValid)
		perm = np.random.choice(self.index, size = self.amtValid)
		self.validIMGs 		= self.validIMGs[perm,:,:]
		self.validLabels 	= self.validLabels[perm,:]

data = Data()

#Data Info
IMG_H 	= data.imgH 		#Image Height
IMG_W 	= data.imgW 		#Image Width
trSamp 	= data.amtTrain		#Number of training samples	
valSamp = data.amtValid		#Number of validation samples

#Convolutional Neural Net Structure
CL_1 	= 32				#Size of Convolutional Layer 1, layer 1
CL_2 	= 64				#Size of Convolutional Layer 2, layer 2
FC_1 	= 128 				#Size of Fully Conected Layer,	layer 3
CLASSES = data.classes 		#Size of Output Layer
WS_1 = 3					#Window Size for CL_1
WS_2 = 5					#Window Size for CL_2
STRIDE = 1

#Training and Hyperparameters
DROP 	= 0.75				#Dropout Keep Probability
LR 		= 1e-3				#Learning Rate
EPOCHS 	= 20				#Go through all the test data this many times
NUM_BATCHES = 10000			#Number of batches per epoch
BATCH_SIZE  = trSamp/NUM_BATCHES	#Number of samples per batch

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


weights = {
	'wCL_1': tf.Variable(tf.random_normal([WS_1, WS_1, 1, CL_1], stddev=0.1)),
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

imgs  	= tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 1])
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
for epoch in range(1):#EPOCHS):
	data.permuteTrain()
	imgs_all = np.expand_dims(data.trainIMGs, axis = 3)
	labels_all = data.trainLabels

	for batch_num in range(NUM_BATCHES):
		ran = np.int32(np.arange(batch_num*BATCH_SIZE, (batch_num+1)*BATCH_SIZE))

		#Current batch
		imgs_curr = imgs_all[ran,:,:,:]
		labels_curr = labels_all[ran,:]

		#Train on batch
		loss_tr, _ = sess.run([loss,optim],\
			feed_dict = {imgs:imgs_curr, labels: labels_curr, dropout: DROP})

		#Check validation data and save parameters
		if (batch_num + 1) % 10 == 0:
			saver.save(sess, checkpoint_dir + 'model_' + str(savePt) + '.ckpt')
			savePt += 1

			data.permuteValid()
			imgs_val = np.expand_dims(data.validIMGs, axis = 3)
			labels_val = data.validLabels

			loss_val, acc_val = sess.run([loss,acc],\
				feed_dict = {imgs: imgs_val, labels: labels_val, dropout: 1})

			print("Epoch " + str(epoch) + " Batch " + str(batch_num + 1) +\
				", Training: Loss = " + "{:4f}".format(loss_tr) + ", Validation: Loss = " +\
				"{.4f}".format(loss_val) + ", Accuracy = " + "{.4f}".format(acc_val))

	print('\n')

print("Training Done")

if acc_val > .0:
	imgs_test = np.expand_dims(data.testIMGs, axis = 3)
	acc_test = sess.run(acc, feed_dict = {imgs: imgs_test, labels: data.testLabels,\
		dropout: 1})
	print("Test Accuracy = " + "{:4f}".format(acc_test))

