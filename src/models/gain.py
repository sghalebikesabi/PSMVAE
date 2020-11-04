'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
                     Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  

def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)
      


def gain (data_x, data_x_test, gain_parameters, args):
    '''Impute missing values in data_x
    
    Args:
        - data_x: original data with missing values
        - gain_parameters: GAIN network parameters:
            - batch_size: Batch size
            - hint_rate: Hint rate
            - alpha: Hyperparameter
            - iterations: Iterations
            
    Returns:
        - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1-np.isnan(data_x)
    data_m_test = 1-np.isnan(data_x_test)
    
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']*int(data_x.shape[0]/gain_parameters['batch_size'])
    
    # Other parameters
    no, dim = data_x.shape
    no_test,_ = data_x_test.shape
    
    # Hidden state dimensions
    h_dim = int(dim)
    
    # Normalization
    norm_data, norm_parameters = normalization(data_x, None, 'minmax')
    norm_data_x = np.nan_to_num(norm_data, 0)
    norm_data_test, norm_parameters_test = normalization(data_x_test, norm_parameters, 'minmax')
    norm_data_x_test = np.nan_to_num(norm_data_test, 0)
    
    norm_data_x = np.nan_to_num(data_x, 0)
    norm_data_x_test = np.nan_to_num(data_x_test, 0)


    ## GAIN architecture     
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [None, dim])
    # Mask vector 
    M = tf.placeholder(tf.float32, shape = [None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [None, dim])
    
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
    # D_W1 = tf.Variable(tf.keras.initializers.glorot_normal()([dim*2, h_dim])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [dim]))    # Multi-variate outputs
    
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    
    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))    
    G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [dim]))
    
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    
    ## GAIN functions
    # Generator
    def generator(x,m):
        # Concatenate Mask and Data
        inputs = tf.concat(values = [x, m], axis = 1) 
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)     
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
        return G_prob
            
    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values = [x, h], axis = 1) 
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)    
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob
    
    ## GAIN structure
    # Generator
    G_sample = generator(X, M)
 
    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)
    
    # Discriminator
    D_prob = discriminator(Hat_X, H)
    
    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                                                + (1-M) * tf.log(1. - D_prob + 1e-8))     
    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
    
    MSE_loss = \
    tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
    
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss 
    
    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    
    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
     
    # Start Iterations
    for it in tqdm(range(iterations)):
        #print("bl")
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]    
        M_mb = data_m[batch_idx, :]    
        # Sample random vectors    
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp
            
        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
            
        _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                                            feed_dict = {M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = \
        sess.run([G_solver, G_loss_temp, MSE_loss],
                         feed_dict = {X: X_mb, M: M_mb, H: H_mb})
                        
    # Return imputed data  
    imputed_list = []          
    for i in range(gain_parameters['num_samples']):
      Z_mb = uniform_sampler(0, 0.01, no, dim)
      M_mb = data_m
      X_mb = norm_data_x                    
      X_mb = (1-args.post_sample) * M_mb * X_mb + args.post_sample * ((1-M_mb) * Z_mb)

      imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0] 

      imputed_data = renormalization(imputed_data, norm_parameters, 'minmax')

      imputed_list.append(imputed_data)
    

    # Return test imputed data            
    imputed_list_test = []          
    for i in range(gain_parameters['num_samples']):
      Z_mb_test = uniform_sampler(0, 0.01, no_test, dim) 
      M_mb_test = data_m_test
      X_mb_test = norm_data_x_test                    
      X_mb_test = (1-args.post_sample) * M_mb_test * X_mb_test + args.post_sample * ((1-M_mb_test) * Z_mb_test)

      imputed_data_test = sess.run([G_sample], feed_dict = {X: X_mb_test, M: M_mb_test})[0]

      imputed_data = renormalization(imputed_data_test, norm_parameters, 'minmax')

      imputed_list_test.append(imputed_data_test)

                    
    return imputed_list, imputed_list_test


