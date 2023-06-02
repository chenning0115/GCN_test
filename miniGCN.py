
"""
@author: danfeng
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hdf5storage as scio 
import scipy.io as sio
from tf_utils import random_mini_batches_GCN
from tensorflow.python.framework import ops
import scipy 
import h5py

class_num = 16
spectral = 103

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def create_placeholders(n_x, n_y):

    isTraining = tf.placeholder_with_default(True, shape=())
    x_in = tf.placeholder(tf.float32,  [None, n_x], name = "x_in")
    y_in = tf.placeholder(tf.float32, [None, n_y], name = "y_in")
    lap_train = tf.placeholder(tf.float32, [None, None], name = "lap_train")
    
    return x_in, y_in, lap_train, isTraining

def initialize_parameters():
   
    tf.set_random_seed(1)

    x_w1 = tf.get_variable("x_w1", [spectral,128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    x_b1 = tf.get_variable("x_b1", [128], initializer = tf.zeros_initializer())

    x_w2 = tf.get_variable("x_w2", [128,class_num], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    x_b2 = tf.get_variable("x_b2", [class_num], initializer = tf.zeros_initializer())    

    
    parameters = {"x_w1": x_w1,
                  "x_b1": x_b1,
                  "x_w2": x_w2,
                  "x_b2": x_b2}
                  
    return parameters

def GCN_layer(x_in, L_, weights):

    x_mid = tf.matmul(x_in, weights)
    x_out = tf.matmul(L_, x_mid)
    
    return x_out

def mynetwork(x, parameters, Lap, isTraining, momentums = 0.9):

    with tf.name_scope("x_layer_1"):

         x_z1_bn = tf.layers.batch_normalization(x, momentum = momentums, training = isTraining)             
         x_z1 = GCN_layer(x_z1_bn, Lap, parameters['x_w1']) + parameters['x_b1']
         x_z1_bn = tf.layers.batch_normalization(x_z1, momentum = momentums, training = isTraining)
         x_a1 = tf.nn.relu(x_z1_bn)      
         
    with tf.name_scope("x_layer_3"):
        
         x_z2_bn = tf.layers.batch_normalization(x_a1, momentum = momentums, training = isTraining)        
         x_z2 = GCN_layer(x_z2_bn, Lap, parameters['x_w2']) + parameters['x_b2']         

    l2_loss =  tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_w2'])
                
    return x_z2, l2_loss

def mynetwork_optimaization(y_est, y_re, l2_loss, reg, learning_rate, global_step):
    
    y_re = tf.squeeze(y_re, name = 'y_re')
    
    with tf.name_scope("cost"):
         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_est, labels = y_re)) +  reg * l2_loss
         
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost,  global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def network_accuracy(x_out, y_in):
    
    correct_prediction = tf.equal(tf.argmax(x_out, 1), tf.argmax(y_in, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
         
    return accuracy
    
def train_mynetwork(x_train, x_test, y_train, y_test, L_train, L_test, learning_rate_base = 0.001, beta_reg = 0.001, num_epochs = 500, minibatch_size = 32, print_cost = True):
    
    ops.reset_default_graph()    
    tf.set_random_seed(1)                          
    seed = 1                                                         
    (m, n_x) = x_train.shape
    (m, n_y) = y_train.shape
    
    costs = []                                        
    costs_dev = []
    train_acc = []
    val_acc = []
    
    x_in, y_in, lap_train, isTraining = create_placeholders(n_x, n_y) 

    parameters = initialize_parameters()
    
    with tf.name_scope("network"):
         x_out, l2_loss = mynetwork(x_in, parameters, lap_train, isTraining)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 50 * m/minibatch_size, 0.5, staircase = True)
    
    with tf.name_scope("optimization"):
         cost, optimizer = mynetwork_optimaization(x_out, y_in, l2_loss, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
         accuracy = network_accuracy(x_out, y_in)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
      
        # Do the training loop
        for epoch in range(num_epochs + 1):
            epoch_cost = 0.
            epoch_acc = 0.
            
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            
            minibatches = random_mini_batches_GCN(x_train, y_train, L_train, minibatch_size, seed)
            
            for minibatch in minibatches:

                # Select a minibatch
                (batch_x, batch_y, batch_l) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x_in: batch_x, y_in: batch_y, lap_train: batch_l, isTraining: True})           
                print("epoch=%s, train_cost=%s, train_acc=%s" % (epoch, minibatch_cost, minibatch_acc))
                epoch_cost += minibatch_cost 
                epoch_acc += minibatch_acc

            epoch_cost_train = epoch_cost / (num_minibatches+ 1) 
            epoch_acc_train = epoch_acc / (num_minibatches+ 1) 

            print("epoch=%s" % epoch)
            if print_cost == True and epoch % 50 == 0:
                features, epoch_cost_test, epoch_acc_test = sess.run([x_out, cost, accuracy], feed_dict={x_in: x_test, y_in: y_test, lap_train: L_test, isTraining: False})
                print ("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_cost_train, epoch_cost_test, epoch_acc_train, epoch_acc_test))
            
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost_train)
                train_acc.append(epoch_acc_train)
                costs_dev.append(epoch_cost_test)
                val_acc.append(epoch_acc_test)

        # plot the cost      
        # plt.plot(np.squeeze(costs))
        # plt.plot(np.squeeze(costs_dev))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()
        
        # plot the accuracy 
        # plt.plot(np.squeeze(train_acc))
        # plt.plot(np.squeeze(val_acc))
        # plt.ylabel('accuracy')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()
      
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
       
       
        return parameters, val_acc, features

# Method 1: uses h5py (WORKS)
def read_sparse(fname):
    f1 = scio.loadmat(fname)
    a = f1['Test_L']
    data,ir,jc = list(a[0])
    M = scipy.sparse.csc_matrix((data, ir, jc))
    cc = M.toarray() 
    return cc


def run_one(sign, sample_num):
    pp ='C:/charnix/codes/vscodes/diffusion_new/hyperclassification/data/miniGCN/'
    prefix = "%s/%s/%s" % (pp, sign, sample_num)
    # prefix = "HSI_GCN"
    Train_X = scio.loadmat('%s/Train_X.mat' % prefix)
    TrLabel = scio.loadmat('%s/TrLabel.mat' % prefix)
    Test_X = scio.loadmat('%s/Test_X.mat' % prefix)
    TeLabel = scio.loadmat('%s/TeLabel.mat' % prefix)
    Train_L = scio.loadmat('%s/Train_L.mat' % prefix)
    # Test_L = scio.loadmat('%s/Test_L.mat' % prefix)
    Test_L = read_sparse('%s/Test_L.mat' % prefix)
    # print(Test_L)

    Train_X = Train_X['Train_X']
    Test_X = Test_X['Test_X']
    TrLabel = TrLabel['TrLabel'].astype(np.int8)
    TeLabel = TeLabel['TeLabel'].astype(np.int8)

    Train_L = Train_L['Train_L']
    # Test_L = Test_L['Test_L']
    print(Test_L.shape, Test_L[0].shape, Test_L.dtype, Test_L[0].dtype, Test_L.size)
    TrLabel = convert_to_one_hot(TrLabel-1, class_num)
    TrLabel = TrLabel.T
    TeLabel = convert_to_one_hot(TeLabel-1, class_num)   
    TeLabel = TeLabel.T

    
    print(Train_X.shape, Test_X.shape, Train_L.shape, Test_L.shape, TeLabel.shape)
    print(Train_X.dtype, Test_X.dtype, Train_L.dtype, Test_L.dtype, TeLabel.dtype)
    print('chenning', type(Test_X), type(TeLabel), type(Test_L))
    parameters, val_acc, features = train_mynetwork(Train_X, Test_X, TrLabel, TeLabel, Train_L, Test_L)
    sio.savemat('%s/features.mat' % prefix, {'features': features})

signtonum = {
    'Indian' : 16,
    'Pavia' : 9,
    'Salinas': 16
}

signtospe = {
    'Indian' : 200,
    'Pavia' : 103,
    'Salinas': 204
}

for s in ['Salinas']:
    spectral = signtospe[s]
    class_num = signtonum[s]
    for n in [10, 20, 30, 40, 50, 60, 70, 80]:
        run_one(s, n)