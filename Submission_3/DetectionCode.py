import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from /tmp/data/ import input_data

#Here one hot means [1,0,0,0,0,0] like structure
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#Hidden Layer 1 
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_class = 10

#Data to be taken at a time 
batch_size = 1000

# height x weight

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):

	#Assign Weights and biases
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer   = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_class])),
					 'biases':tf.Variable(tf.random_normal([n_class]))}

	#Forward Propogation 				 
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])				 
 	#Kind of thresholding function 
 	l1 = tf.nn.relu(l1)			 
			 
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])		 
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])		 
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights'])+output_layer['biases'] 

	# This is the output from deep neural network 
	return output

def train_neural_network(x):
	#Neural Network
	prediction = neural_network_model(x)
	#Cost function defined
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	#Optimizer for back propogation 
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#No of times you want to do back run back propogations 
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range (hm_epochs):
			epoch_loss = 0;
			
			pl = int(mnist.train.num_examples/batch_size)

			for _ in range (pl):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer,cost],feed_dict= {x:epoch_x,y:epoch_y})
				epoch_loss += c
			print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))	
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)