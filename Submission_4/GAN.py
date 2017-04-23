#Imports
import tensorflow as tf		#	Tensorflow for NN
import random			#	For generating Random Numbers
from random import randint
import numpy as np		#	For Numbers
import matplotlib.pyplot as plt	#	For Ploting Data
import skimage
from numpy.linalg import inv
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = read_data_sets("Faces/")
from os import listdir
from PIL import Image
from numpy import *
from scipy import misc
import glob

def em_PPCA(t, k):
    if k < 1 | k > t.shape[1]:
        print('Number of Principle Components must be integer, >0, <dim')

    # Number of iteration.
    ite = 25

    # Finding the height and the width of the data matrix.
    [height, width] = t.shape
    
    t_mean = np.zeros((height,1))
    # Finding the mean value of the observed data vectors.
    # t_mean = (sum(t(:,1:width)')')/width;
    # print(t[:, 0:width].shape)
    for i in range(height):
        t_mean[i] = (np.sum(t[i,:].T).T) / width

    # t_mean = 0
    # for i in range(width)
    #     t_mean[i] = (np.sum(t[:, i].T).T) / width

    # Normalize the data matrix.

    t = t - np.dot((t_mean) , np.ones((1, width)))
    # print(t)
    # Initially w and sigma square will be randomly selected.

    W = np.random.standard_normal((height, k))
    #print(W.shape)
    sigma_square = np.random.standard_normal((1, 1))
    #print(sigma_square)

    print('EM algorithm is running....Please Wait.......')

    for i in range(ite):
    	print 'EM - iter : -',i
        #print(W)
        #print(sigma_square)
        # According to the equation: M = W'W + Sigma^2*I
        M = np.dot(W.T,W) + (sigma_square * np.eye(k, k))

        # Find inverse of M
        inv_M = inv(M)

        # Expected Xn
        Xn = np.zeros((k, width))
        Xn_Xn_T = np.zeros((k, k))

        for i in range(width):
            Xn[:, i] = np.dot(np.dot((inv_M), W.T) ,t[:,i])
            # Find Expected of XnXn'
            Xn_Xn_T = Xn_Xn_T + (sigma_square * (inv_M) + np.dot( (Xn[:, i].reshape(len(Xn),1)) , (Xn[:, i].reshape(len(Xn),1).T)))

        #print(Xn)
        # Taking the old value of W
        old_W = W

        temp1 = np.zeros((height, k))

        # print(t.shape)

        for i in range(width):

            temp1 = temp1 + np.dot(t[:,i].reshape(height,1) , ((Xn[:,i]).reshape(len(Xn),1).T))

            # Taking the new value of W
        W = np.dot(temp1 , inv(Xn_Xn_T))

        #print(W)
        sum11 = 0

        for i in range(width):

            temp2 = sigma_square * inv_M + np.dot( (Xn[:,i].reshape(len(Xn),1)) , (Xn[:,i].reshape(len(Xn),1).T) )
            sum11 = sum11 + ((np.linalg.norm(t[:,1]) ** 2) - np.dot(np.dot((2 * (Xn[:,i].reshape(len(Xn),1).T)) ,(W.T)) , (t[:,i].reshape(height,1)) ) + np.trace((np.dot(np.dot(temp2 , (W.T)) , W))))

        #print(temp2)
        sigma_square = sum11 / (width * height)

    print('EM Algorithm Completed. W is created. Press enter to continue')
    print(sigma_square)
    M = np.dot((W.T) , W) + sigma_square * np.eye(k,k)
    In_M = inv(M)

    Xn = np.dot(np.dot((In_M),(W.T)) , t)

    print('Principal Component are ready.. press enter to continue')
    # print(sigma_square)
    # print(W)
    return W, sigma_square, Xn, t_mean, M               #W-ppca Xn

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
		img = Image.open(path + image).convert('LA')		# Opening Image and conveting into Grayscale image
		#img = Image.open(path + image) # open colour image
		#img = img.convert('1') # convert image to black and white
#		image_file.save('result.png')
		loadedImages.append(img)
		
    return loadedImages

path = "/home/jay/Desktop/mnist_png/training/0/"

# your images in an array
imgs = loadImages(path)
#imgs=[]
#for i in imgs1:
#	tmp = np.asarray(i)
#	imgs.append(tmp)

print 'aaaaaaa : ',len(imgs)

# Isolating Images
# x_train = mnist.train.images[:450,:]
# x_train.shape

# Let's look at what a random image might look like.
randomNum = random.randint(0,5923)
#image = imgs[randomNum].reshape([896,592])
image = imgs[randomNum]
#image = rescale_intensity(image, out_range=(0, 255))
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()


# Defining Functions in order to helping with CNN.
def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolution Layer and Passing Image into it for discrimination.

def discriminator(image, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    #First Conv and Pool Layers
    W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    h_pool1 = avg_pool_2x2(h_conv1)

    #Second Conv and Pool Layers
    W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = avg_pool_2x2(h_conv2)

    #First Fully Connected Layer
    W_fc1 = tf.get_variable('d_wfc1', [28*28, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
    h_pool2_flat = tf.reshape(h_pool2, [-1,28*28])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Second Fully Connected Layer
    W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

    #Final Layer
    y_conv=(tf.matmul(h_fc1, W_fc2) + b_fc2)
    return y_conv

def generator(z, batch_size, z_dim,required,mean_X,reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    g_dim = 64  #Number of filters of first layer of generator 
    c_dim = 1   #Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
    s = 24 	#Output size of the image
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) #We want to slowly upscale the image, so these values will help
                                                              #make that change gradual.

    h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
    h0 = tf.nn.relu(h0)
    #Dimensions of h0 = batch_size x 2 x 2 x 25

    #First DeConv Layer
    output1_shape = [batch_size, s8, s8, g_dim*4]
    W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])], 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
    H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], padding='SAME')
    H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
    H_conv1 = tf.nn.relu(H_conv1)
    #Dimensions of H_conv1 = batch_size x 3 x 3 x 256

    #Second DeConv Layer
    output2_shape = [batch_size, s4, s4, g_dim*2]
    W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])], 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
    H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], padding='SAME')
    H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
    H_conv2 = tf.nn.relu(H_conv2)
    #Dimensions of H_conv2 = batch_size x 6 x 6 x 128

    #Third DeConv Layer
    output3_shape = [batch_size, s2, s2, g_dim*1]
    W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])], 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
    H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME')
    H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
    H_conv3 = tf.nn.relu(H_conv3)
    #Dimensions of H_conv3 = batch_size x 12 x 12 x 64

    #Fourth DeConv Layer
    output4_shape = [batch_size, s, s, c_dim]
    W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])], 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
    H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], padding='SAME')
    H_conv4 = tf.nn.tanh(H_conv4)
    #Dimensions of H_conv4 = batch_size x 24 x 24 x 1
    
    Result = tf.reshape(H_conv4,[batch_size,576,1,1])    
    final = tf.zeros([0,28,28,1],tf.float32)
    required = tf.reshape(required,[784,576,1])
    for i in range(batch_size):
    	TEMP = tf.reshape(tf.reduce_sum(required[:,:,None]*Result[i],1),[1,28,28,1])
    	final = tf.concat([final,TEMP],0)
    
    return final

data = np.empty([784,0]);
for image_path in glob.glob("/home/jay/Desktop/Images/*.png"):
    #image = misc.imread(image_path)
    #image = Image.fromarray(resizelist[image_path])
    mat = asarray(Image.open(image_path))
    mat = mat.reshape([784,1]);
    data = np.append(data,mat,1);
    #print 'V is : \n',V.shape
    #print 'S is : \n',S.shape
    #print 'mean is : \n',mean_X.shape


print data.shape
[W,sigma_square,Xn,t_mean,M] = em_PPCA(data,576)

data = data.transpose()
#[V_Get,S,mean_X] = pca(data);
#ppca = PPCA(data)
#required = ppca.fit(d=576, verbose=True)
#required = V[:,:576]

W = W.astype('float32')
t_mean = t_mean.astype('float32')

sess = tf.Session()
z_dimensions = 100
z_test_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])


sample_image = generator(z_test_placeholder, 1, z_dimensions,W,t_mean)
test_z = np.random.normal(-1, 1, [1,z_dimensions])

sess.run(tf.global_variables_initializer())
temp = (sess.run(sample_image, feed_dict={z_test_placeholder: test_z}))

#	Finally, we can view the output through matplotlib.
my_i = temp.squeeze()
plt.imshow(my_i, cmap='gray_r')
plt.show()

batch_size = 16
tf.reset_default_graph() #Since we changed our batch size (from 1 to 16), we need to reset our Tensorflow graph

sess = tf.Session()
x_placeholder = tf.placeholder("float", shape = [None,28,28,1]) #Placeholder for input images to the discriminator
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions]) #Placeholder for input noise vectors to the generator

Dx = discriminator(x_placeholder) #Dx will hold discriminator prediction probabilities for the real MNIST images
Gz = generator(z_placeholder, batch_size, z_dimensions,W,t_mean) #Gz holds the generated images
Dg = discriminator(Gz, reuse=True) #Dg will hold discriminator prediction probabilities for generated images

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.ones_like(Dg)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.zeros_like(Dg)))

d_loss = d_loss_real + d_loss_fake

#g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg, tf.ones_like(Dg)))
#d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dx, tf.ones_like(Dx)))
#d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dg, tf.zeros_like(Dg)))

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

with tf.variable_scope(tf.get_variable_scope(),reuse=False):
	trainerD = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
	trainerG = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

sess.run(tf.global_variables_initializer())
iterations = 12500
num = 0;
for i in range(iterations):
	z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])
    #	real_image_batch = mnist.train.next_batch(batch_size)
    #	real_image_batch = np.reshape(real_image_batch[0],[batch_size,28,28,1])
    #for img in imgs:
	real_image_batch = np.empty([batch_size,28,28,1],dtype='float32')
	for j in range(batch_size):
		real_image_batch[j,:,:,:] = np.reshape(data[randint(0,979),:],[1,28,28,1]) 
	#real_image_batch1 = np.delete(real_image_batch[:][:][:],[1])
	#print 'adadad : ',real_image_batch[0]
	#print 'ssss : ',(real_image_batch[0][0][0])
	#temp=[]
	print  num
	num = num + 1 	
	#for j in xrange(len(real_image_batch)):
#		temp1=[]
#		for k in xrange(len(real_image_batch[j])):
#			temp2=[]
#			for l in xrange(len(real_image_batch[j][k])):
#				#print real_image_batch[j][k][l]
#				temp2.append(np.delete(real_image_batch[j][k][l],[1]))
#				#print real_image_batch[j][k][l]
#			temp1.append(temp2)
#		temp.append(temp1)
	#print len(temp[0][0])
    #real_image_batch = imgs[:batch_size];
	_,dLoss = sess.run([trainerD, d_loss],feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch}) #Update the discriminator
	_,gLoss = sess.run([trainerG,g_loss],feed_dict={z_placeholder:z_batch}) #Update the generator

sample_image = generator(z_placeholder, 1, z_dimensions,W,t_mean)
z_batch = np.random.normal(-1, 1, size=[1, z_dimensions])
temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i, cmap='gray_r')
plt.show()
