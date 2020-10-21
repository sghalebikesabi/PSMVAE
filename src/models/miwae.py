# TODO delete this file before submission


#!/usr/bin/env python
# coding: utf-8

# # not-MIWAE: Deep Generative Modelling with Missing not at Random Data
# This notebook illustrates how to fit a *deep latent variable model* to data affected by a missing process which depends on the missing data itself, i.e. *missing not at random*.
# 
# We fit a linear PPCA-like model to a relatively small UCI dataset.

# ### Preamble

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import keras
import matplotlib.pyplot as plt
import pandas as pd
import time
from utils import rmse_loss


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 15.0
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['lines.linewidth'] = 2.5

from utils import renormalization, rounding

# ### Load data
# Here we use the white-wine dataset from the UCI database

# In[2]:


# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
# data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
# # ---- drop the classification attribute
# data = data[:, :-1]


# ### Settings

# In[3]:




# ### Standardize data

# In[4]:

def notMiwae(compl_data_train, data_train, compl_data_test, data_test, norm_parameters, wandb, args):
    
    N, D = data_train.shape
    n_latent = D - 1
    n_hidden = 128
    n_samples = args.num_samples_train
    max_iter = int(args.max_epochs*N/args.batch_size)
    batch_size = args.batch_size
    
    compl_data_train_renorm = renormalization(compl_data_train.copy(), norm_parameters)

    # # ---- standardize data
    # data = data - np.mean(data, axis=0)
    # data = data / np.std(data, axis=0)

    # # ---- random permutation 
    # p = np.random.permutation(N)
    # data = data[p, :]

    # ---- we use the full dataset for training here, but you can make a train-val split
    Xtrain = compl_data_train#.values
    Xval = compl_data_test#.values


    # ### Introduce missing 
    # Here we denote
    # - Xnan: data matrix with np.nan as the missing entries
    # - Xz: data matrix with 0 as the missing entries
    # - S: missing mask 
    # 
    # The missing process depends on the missing data itself:
    # - in half the features, set the feature value to missing when it is higher than the feature mean

    # In[5]:


    # ---- introduce missing process
    # Xnan = Xtrain.copy()
    # Xz = Xtrain.copy()

    # mean = np.mean(Xnan[:, :int(D / 2)], axis=0)
    # ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean

    # Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan
    # Xz[:, :int(D / 2)][ix_larger_than_mean] = 0

    # S = np.array(~np.isnan(Xnan), dtype=np.float32)

    Xnan = data_train#.values
    Xnan_test = data_test#.values
    S = ~np.isnan(data_train)#.values
    Xz = data_train.copy()
    Xz[Xz!=Xz] = 0


    # ### Build the model
    # The model we are building has a Gaussian prior and a Gaussian observation model,
    # 
    # $$ p(\mathbf{z}) = \mathcal{N}(\mathbf{z} | \mathbf{0}, \mathbf{I})$$
    # 
    # $$ p(\mathbf{x} | \mathbf{z}) = \mathcal{N}(\mathbf{x} | \mathbf{\mu}_{\theta}(\mathbf{z}), \sigma^2\mathbf{I})$$
    # 
    # $$ p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z})p(\mathbf{z}) d\mathbf{z}$$
    # 
    # where $\mathbf{\mu}_{\theta}(\mathbf{z}): \mathbb{R}^d \rightarrow \mathbb{R}^p $ in general is a deep neural net, but in this case is a linear mapping, $\mathbf{\mu}_{\theta}(\mathbf{z}) = \mathbf{Wz + b}$.
    # 
    # The variational posterior is also Gaussian
    # 
    # $$q_{\gamma}(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mathbf{z} | \mu_{\gamma}(\mathbf{x}), \sigma_{\gamma}(\mathbf{x})^2 \mathbf{I})$$
    # 
    # If the missing process is *missing at random*, it is ignorable and the ELBO becomes, as described in [the MIWAE paper](https://arxiv.org/abs/1812.02633)
    # 
    # $$ E_{\mathbf{z}_1...\mathbf{z}_K} \left[ \log \frac{1}{K}\sum_{k=1}^K \frac{p_{\theta}(\mathbf{x^o} | \mathbf{z}_k)p(\mathbf{z}_k)}{q_{\gamma}(\mathbf{z}_k | \mathbf{x^o})} \right] $$
    # 
    # When the missing process is MNAR it is non-ignorable and we need to include the missing model. In this example we include the missing model as a logistic regression in each feature
    # 
    # $$ p_{\phi}(\mathbf{s} | \mathbf{x^o, x^m}) = \text{Bern}(\mathbf{s} | \pi_{\phi}(\mathbf{x^o, x^m}))$$
    # 
    # $$ \pi_{\phi, j}(x_j) = \frac{1}{1 + e^{-\text{logits}_j}} $$
    # 
    # $$ \text{logits}_j = W_j (x_j - b_j) $$
    # 
    # The ELBO in the MNAR case becomes
    # 
    # $$ E_{(\mathbf{z}_1, \mathbf{x}_1^m)...(\mathbf{z}_K, \mathbf{x}_K^m)} \left[ \log \frac{1}{K} \sum_{k=1}^K \frac{p_{\phi}(\mathbf{s} | \mathbf{x}^o, \mathbf{x}_k^m) p_{\theta}(\mathbf{x}^o | \mathbf{z}_k) p(\mathbf{z}_k)}{q_{\gamma}(\mathbf{z} | \mathbf{x}^o)} \right]$$
    # 

    # ### Inputs
    # Let's first define the inputs of the model
    # - x_pl: data input
    # - s_pl: mask input
    # - n_pl: number of importance samples

    # In[10]:


    print("Creating graph...")
    # tf.reset_default_graph()
    # ---- input
    with tf.compat.v1.variable_scope('input'):
        tf.compat.v1.disable_eager_execution()
        x_pl = tf.compat.v1.placeholder(tf.float32, [None, D], 'x_pl')
        s_pl = tf.compat.v1.placeholder(tf.float32, [None, D], 's_pl')
        n_pl = tf.compat.v1.placeholder(tf.int32, shape=(), name='n_pl')


    # the noise variance is learned as a shared parameter

    # In[13]:


    # ---- parameters
    with tf.compat.v1.variable_scope('data_process'):
        logstd = tf.compat.v1.get_variable('logstd', shape=[])


    # ### Encoder
    # The encoder / inference network consists of two hidden layers with 128 units and tanh activation

    # In[14]:


    x = keras.layers.Dense(units=n_hidden, activation=tf.nn.tanh, name='l_enc1')(x_pl)
    x = keras.layers.Dense(units=n_hidden, activation=tf.nn.tanh, name='l_enc2')(x)

    q_mu = keras.layers.Dense(units=n_latent, activation=None, name='q_mu')(x)

    q_logstd = keras.layers.Dense(units=n_latent, activation=lambda x: tf.clip_by_value(x, -10, 10),
                            name='q_logstd')(x)


    # ### Variational distribution

    # In[15]:


    q_z = tfp.distributions.Normal(loc=q_mu, scale=tf.exp(q_logstd))

    # ---- sample the latent value
    l_z = q_z.sample(n_pl)                    # shape [n_samples, batch_size, dl]
    l_z = tf.transpose(l_z, perm=[1, 0, 2])   # shape [batch_size, n_samples, dl]


    # ### Decoder

    # In[16]:


    mu = keras.layers.Dense(units=D, activation=None, name='mu')(l_z)


    # ### Observation model / likelihood function

    # In[17]:


    p_x_given_z = tfp.distributions.Normal(loc=mu, scale=tf.exp(logstd))


    # ### Missing model
    # - first mix observed data and samples of missing data
    # - feed through missing model
    # - find likelihood of missing model parameters
    # 
    # We have to expand the dimensions of x_pl and s_pl, since mu has size [batch, n_samples, D]

    # In[18]:


    l_out_mixed = mu * tf.expand_dims(1 - s_pl, axis=1) + tf.expand_dims(x_pl * s_pl, axis=1)


    # In[20]:


    W = tf.compat.v1.get_variable('W', shape=[1, 1, D])
    W = -tf.nn.softplus(W)
    b = tf.compat.v1.get_variable('b', shape=[1, 1, D])

    logits = W * (l_out_mixed - b)

    p_s_given_x = tfp.distributions.Bernoulli(logits=logits)


    # ### Evaluating likelihoods

    # In[21]:


    # ---- evaluate the observed data in p(x|z)
    log_p_x_given_z = tf.reduce_sum(tf.expand_dims(s_pl, axis=1) * 
                                    p_x_given_z.log_prob(tf.expand_dims(x_pl, axis=1)), axis=-1)  

    # --- evaluate the z-samples in q(z|x)
    q_z2 = tfp.distributions.Normal(loc=tf.expand_dims(q_z.loc, axis=1), scale=tf.expand_dims(q_z.scale, axis=1))
    log_q_z_given_x = tf.reduce_sum(q_z2.log_prob(l_z), axis=-1)

    # ---- evaluate the z-samples in the prior p(z)
    prior = tfp.distributions.Normal(loc=0.0, scale=1.0)
    log_p_z = tf.reduce_sum(prior.log_prob(l_z), axis=-1)

    # ---- evaluate the mask in p(s|x)
    log_p_s_given_x = tf.reduce_sum(p_s_given_x.log_prob(tf.expand_dims(s_pl, axis=1)), axis=-1)


    # ### Losses for the MIWAE and not-MIWAE respectively

    # In[24]:


    lpxz = log_p_x_given_z
    lpz = log_p_z
    lqzx = log_q_z_given_x
    lpsx = log_p_s_given_x

    # ---- MIWAE
    # ---- importance weights
    l_w = lpxz + lpz - lqzx

    # ---- sum over samples
    log_sum_w = tf.reduce_logsumexp(l_w, axis=1)

    # ---- average over samples
    log_avg_weight = log_sum_w - tf.compat.v1.log(tf.cast(n_pl, tf.float32))

    # ---- average over minibatch to get the average llh
    MIWAE = tf.reduce_mean(log_avg_weight, axis=-1)


    # ---- not-MIWAE
    # ---- importance weights
    l_w = lpxz + lpsx + lpz - lqzx

    # ---- sum over samples
    log_sum_w = tf.reduce_logsumexp(l_w, axis=1)

    # ---- average over samples
    log_avg_weight = log_sum_w - tf.compat.v1.log(tf.cast(n_pl, tf.float32))

    # ---- average over minibatch to get the average llh
    notMIWAE = tf.reduce_mean(log_avg_weight, axis=-1)


    # ### Training stuff

    # In[28]:


    # ---- training stuff
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    global_step = tf.Variable(initial_value=0, trainable=False)
    optimizer = tf.compat.v1.train.AdamOptimizer()


    # ### Choose wether you want to train the MIWAE or the notMIWAE

    # In[31]:


    if args.model_class == 'notmiwae':
        loss = -notMIWAE
    elif args.model_class == 'miwae':
        loss = -MIWAE

    tvars = tf.compat.v1.trainable_variables()
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=tvars)
    sess.run(tf.compat.v1.global_variables_initializer())


    # ### Do the training

    # In[32]:
    # ---- single imputation in the MIWAE
    def imputationRMSE(sess, Xorg, Xnan, L):

        N = len(Xorg)
        
        Xz = Xnan.copy()
        Xz[np.isnan(Xnan)] = 0
        S = np.array(~np.isnan(Xnan), dtype=np.float32)

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1)[:, None])
            return e_x / e_x.sum(axis=1)[:, None]

        def imp(xz, s, L):
            _mu, _log_p_x_given_z, _log_p_z, _log_q_z_given_x = sess.run(
                [mu, log_p_x_given_z, log_p_z, log_q_z_given_x],
                {x_pl: xz, s_pl: s, n_pl: L})

            wl = softmax(_log_p_x_given_z + _log_p_z - _log_q_z_given_x)

            xm = np.sum((_mu.T * wl.T).T, axis=1)
            xmix = xz + xm * (1 - s)

            return _mu, wl, xm, xmix

        XM = np.zeros_like(Xorg)

        for i in range(N):

            xz = Xz[i, :][None, :]
            s = S[i, :][None, :]

            _mu, wl, xm, xmix = imp(xz, s, L)

            XM[i, :] = xm

            # if i % 100 == 0:
            #     print('{0} / {1}'.format(i, N))

        return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XM


    # ---- single imputation in the not-MIWAE
    def not_imputationRMSE(sess, Xorg, Xnan, L):

        N = len(Xorg)
        
        Xz = Xnan.copy()
        Xz[np.isnan(Xnan)] = 0
        S = np.array(~np.isnan(Xnan), dtype=np.float32)

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1)[:, None])
            return e_x / e_x.sum(axis=1)[:, None]

        def imp(xz, s, L):
            _mu, _log_p_x_given_z, _log_p_z, _log_q_z_given_x, _log_p_s_given_x  = sess.run(
                [mu, log_p_x_given_z, log_p_z, log_q_z_given_x, log_p_s_given_x],
                {x_pl: xz, s_pl: s, n_pl: L})

            wl = softmax(_log_p_x_given_z + _log_p_s_given_x + _log_p_z - _log_q_z_given_x)

            xm = np.sum((_mu.T * wl.T).T, axis=1)
            xmix = xz + xm * (1 - s)

            return _mu, wl, xm, xmix

        XM = np.zeros_like(Xorg)

        for i in range(N):

            xz = Xz[i, :][None, :]
            s = S[i, :][None, :]

            _mu, wl, xm, xmix = imp(xz, s, L)

            XM[i, :] = xm

        return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XM


    batch_pointer = 0

    start = time.time()
    best = float("inf")


    for i in range(max_iter):
        x_batch = Xz[batch_pointer: batch_pointer + batch_size, :]
        s_batch = S[batch_pointer: batch_pointer + batch_size, :]

        _, _loss, _step = sess.run([train_op, loss, global_step], {x_pl: x_batch, s_pl: s_batch, n_pl: n_samples})

        batch_pointer += batch_size
        
        if batch_pointer > N - batch_size:
            batch_pointer = 0

            p = np.random.permutation(N)
            Xz = Xz[p, :]
            S = S[p, :]

        if False:
            
            if args.model_class == 'notmiwae':
                rmse, imputations = not_imputationRMSE(sess, Xtrain, Xnan, args.num_samples)
                rmse, imputations_test = not_imputationRMSE(sess, Xval, Xnan_test, 1)
            elif args.model_class == 'miwae':
                rmse, imputations = imputationRMSE(sess, Xtrain, Xnan, args.num_samples)
                rmse, imputations_test = imputationRMSE(sess, Xval, Xnan_test, 1)

            # rounding
            train_imputed = renormalization(imputations, norm_parameters)
            train_imputed = rounding(train_imputed, compl_data_train)

            train_mis_mse = rmse_loss(train_imputed, compl_data_train_renorm, np.isnan(Xnan))

            if train_mis_mse != train_mis_mse:
                raise('NANError')

            wandb.log({'Train Imputation RMSE loss': train_mis_mse})

    if args.model_class == 'notmiwae':
        rmse, imputations = not_imputationRMSE(sess, Xtrain, Xnan, args.num_samples)
        rmse, imputations_test = not_imputationRMSE(sess, Xval, Xnan_test, 1)
    elif args.model_class == 'miwae':
        rmse, imputations = imputationRMSE(sess, Xtrain, Xnan, args.num_samples)
        rmse, imputations_test = imputationRMSE(sess, Xval, Xnan_test, 1)


    return(imputations, imputations_test)
