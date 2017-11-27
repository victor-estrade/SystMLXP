# coding : utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import tempfile

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from .nn import TFClassifier

class _PAN_clasifieur(TFClassifier):
    """PAN"""
    def __init__(self, input_shape=None, n_classes=2, learning_rate=0.05, width=0.01, trade_off=0, model_dir=None, skew=None, preprocessing=None,
                n_steps=1000, n_steps_pre_training=1000, n_steps_catch_training=20, batch_size=128,
                n_layers_D=3, n_neurons_per_layer_D=64, n_layers_R=3, n_neurons_per_layer_R=64, 
                opti_D="Adam", opti_R="SGD", opti_DR="SGD"):
        super(_PAN_clasifieur, self).__init__()
        if input_shape is None:
            raise ValueError("input_shape should be given !")
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.trade_off = trade_off
        self.n_steps = n_steps
        self.n_steps_pre_training = n_steps_pre_training
        self.n_steps_catch_training = n_steps_catch_training
        self.batch_size = batch_size
        self.width = width
        self.scaler = StandardScaler()
        self.skew = skew
        self.preprocessing = preprocessing
        self.n_layers_D = n_layers_D
        self.n_neurons_per_layer_D = n_neurons_per_layer_D
        self.n_layers_R = n_layers_R
        self.n_neurons_per_layer_R = n_neurons_per_layer_R
        self.opti_D = opti_D
        self.opti_R = opti_R
        self.opti_DR = opti_DR
        self.build()
        self._i_run =0
        if model_dir is None:
            self.model_dir = tempfile.mkdtemp()
        else:
            self.model_dir = model_dir
    
    def get_name(self):
        return '{}-{}x{}-{}x{}-{}-{}-{}-{}-{}-{}'.format(self.name, self.n_layers_D, self.n_neurons_per_layer_D, 
                                               self.n_layers_R, self.n_neurons_per_layer_R, 
                                               self.learning_rate, self.trade_off, self.width, 
                                               self.opti_D, self.opti_R, self.opti_DR)

    def save(self, dir_path):
        """Save the model in the given directory"""
        path = os.path.join(dir_path, 'weights.ckpt')
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            saver = tf.train.Saver()
            saver.save(sess, path)
        path = os.path.join(dir_path, 'Scaler.pkl')
        joblib.dump(self.scaler, path)


    def load(self, dir_path):
        """Load the model of th i-th CV from the given directory"""
        path = os.path.join(dir_path, 'weights.ckpt')
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, path)
        path = os.path.join(dir_path, 'Scaler.pkl')
        self.scaler = joblib.load(path)
        return self

    def get_summary_dir(self):
        dir_path =  os.path.join(self.model_dir, self.get_name())
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        print(dir_path)
        return dir_path
        
    def build(self):
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            self.in_X = tf.placeholder(tf.float32, shape=(None,)+self.input_shape, name='in_X')
            self.in_y = tf.placeholder(tf.int32, shape=(None,), name='in_y')
            self.in_z = tf.placeholder(tf.float32, shape=(None,), name='in_z')
            self.in_w = tf.placeholder(tf.float32, shape=(None,), name='in_w')
            y = tf.one_hot(self.in_y, self.n_classes, 1.0, 0.0)#self.in_y
            # y = tf.reshape(y, (-1, 10))
            z = self.in_z#tf.one_hot(self.in_z, 2, 1.0, 0.0)
            
            def stack_softplus_layers(in_, n_layers, n_neurons):
                couche = in_
                for i in range(n_layers):
                    couche = tf.contrib.layers.fully_connected(couche,
                                                               n_neurons,
                                                               activation_fn=tf.nn.softplus)
                return couche
            def stack_relu_layers(in_, n_layers, n_neurons):
                couche = in_
                for i in range(n_layers):
                    couche = tf.contrib.layers.fully_connected(couche,
                                                               n_neurons,
                                                               activation_fn=tf.nn.softplus)
                return couche
            
            with tf.variable_scope('D'):
                
                previous_layer = self.in_X
                for i in range(self.n_layers_D):
                    dense_i = tf.contrib.layers.fully_connected(previous_layer, self.n_neurons_per_layer_D, 
                                            activation_fn=None, scope='dense_{}'.format(i))
                    act_i = tf.nn.softplus(dense_i)
                    previous_layer = act_i
                # self.D_int = stack_softplus_layers(self.in_X, self.n_layers_D, self.n_neurons_per_layer_D)
                # self.D = tf.contrib.layers.fully_connected(self.D_int, 1, activation_fn=tf.identity, scope='couche4')
                self.D = tf.contrib.layers.fully_connected(act_i, self.n_classes, activation_fn=tf.identity, scope='couche4')
                self.output_D = tf.nn.softmax(self.D)
                
            with tf.variable_scope('R'):

                self.R_int = stack_relu_layers(self.output_D, self.n_layers_R, self.n_neurons_per_layer_R)
                self.R = tf.contrib.layers.fully_connected(self.R_int, 1, activation_fn=tf.identity, scope='couche4')
                self.output_R = tf.reshape(self.R, (-1,))
    
            trainable_R = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'R')

            trainable_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D')

            self.op_predict_proba_D = self.output_D
            self.op_predict_D = tf.argmax(self.output_D, axis=1)
            self.op_loss_D = tf.losses.softmax_cross_entropy(y, self.D, weights=self.in_w)
            
            self.op_predict_R = self.output_R
            self.op_loss_R = tf.losses.mean_squared_error(z, self.output_R, weights=self.in_w)
            
            self.op_predict_proba = [self.output_D, self.output_R]
            self.op_predict = [tf.argmax(self.op_predict_proba[0], axis=1), self.op_predict_proba[1]]
            self.op_loss = tf.subtract(self.op_loss_D, self.trade_off * self.op_loss_R)
            
            self.op_train =tf.contrib.layers.optimize_loss(
                loss=self.op_loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.opti_DR, 
                variables=trainable_D)
            
            self.op_train_R = tf.contrib.layers.optimize_loss(
                loss=self.op_loss_R,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.opti_R, 
                variables=trainable_R)
            
            self.op_train_D = tf.contrib.layers.optimize_loss(
                loss=self.op_loss_D,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.opti_D, 
                variables=trainable_D)
            self.op_init = tf.global_variables_initializer()
            
            # Summaries
            tf.summary.scalar('Classification_Loss', self.op_loss_D)
            tf.summary.scalar('Regressor_Loss', self.op_loss_R)
            tf.summary.scalar('DR_Loss', self.op_loss)
            
        self.init()
        
    def predict_D(self, X):
        if self.preprocessing is not None:
            X, y, sample_weight = self.preprocessing(X, None, None, training=True)
        X = self.scaler.transform(X)
        return self.sess.run(self.op_predict_D, feed_dict={self.in_X:X})
    
    def predict_proba_D(self, X):
        if self.preprocessing is not None:
            X, y, sample_weight = self.preprocessing(X, None, None, training=True)
        X = self.scaler.transform(X)
        return self.sess.run(self.op_predict_proba_D, feed_dict={self.in_X:X})
    
    def predict_proba(self, X, colum=True):
        return self.predict_proba_D(X)

    def predict(self, X):
        return self.predict_D(X).flatten()

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def fit_DR(self, generator, n_steps=1):
        batch_gen = generator
        batch_size = self.batch_size
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            for i in range(n_steps):
                X_batch, y_batch, z_batch, w_batch = next(batch_gen)
                feed_dict = {self.in_X: X_batch, 
                            self.in_z: z_batch.reshape(batch_size, ), 
                            self.in_y: y_batch.reshape(batch_size, ),
                            self.in_w: w_batch.reshape(batch_size, )}
                loss = sess.run(self.op_train,
                                feed_dict=feed_dict)
                
                if i%20 ==0 :
                    summary_str = sess.run(self.summaries, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, i)
                    self.summary_writer.flush()

        return self
    
    def fit_R(self, generator, n_steps=1):
        batch_gen = generator
        batch_size = self.batch_size
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            for i in range(n_steps):
                X_batch, y_batch, z_batch, w_batch = next(batch_gen)
                feed_dict = {self.in_X: X_batch, 
                            self.in_z: z_batch.reshape(batch_size, ), 
                            self.in_y: y_batch.reshape(batch_size, ),
                            self.in_w: w_batch.reshape(batch_size, )}
                loss = sess.run(self.op_train_R,
                                feed_dict=feed_dict)

        return self
    
    def fit_D(self, generator, n_steps=1):
        batch_gen = generator
        batch_size = self.batch_size
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            for i in range(n_steps):
                X_batch, y_batch, z_batch, w_batch = next(batch_gen)
                feed_dict = {self.in_X: X_batch, 
                            self.in_z: z_batch.reshape(batch_size, ), 
                            self.in_y: y_batch.reshape(batch_size, ),
                            self.in_w: w_batch.reshape(batch_size, )}
                loss = self.sess.run(self.op_train_D,
                                feed_dict=feed_dict)
                
                if i%20 ==0 :
                    summary_str = sess.run(self.summaries, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, i)
                    self.summary_writer.flush()
                    
        return self
    
    def fit(self, X, y, z=None, sample_weight=None):
        """ pré-entraine D puis plusieur fois R et D"""
            
        if sample_weight is None:
            sample_weight = np.ones_like(y)

        self.reset()
        self.init()
        self._i_run +=1
        
        if z is None:
            if self.skew is None:
                raise ValueError("skew ou z manquant")
            z = self.sample_z(X, self.width)
        X = self.skew(X, z)
        
        if self.preprocessing is not None:
            X, y, sample_weight = self.preprocessing(X, y, sample_weight, training=True)
        X = self.scaler.fit_transform(X)
        X = np.asarray(X)
        y = np.asarray(y)
        w = np.asarray(sample_weight)
        
        generator_R = self.minibatch_generator(X, y, z, w, batch_size=self.batch_size)
        generator_DR = self.minibatch_generator(X, y, z, w, batch_size=self.batch_size)
        
        print("starting for trade_off={}".format(self.trade_off))
        
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            self.summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.get_summary_dir(), sess.graph)

        self.fit_D(generator_R, n_steps=self.n_steps_pre_training)

        if (self.trade_off != 0) :
            self.fit_R(generator_R, n_steps=self.n_steps_pre_training)
            for i in range(self.n_steps):
                self.fit_DR(generator_DR, n_steps=1)
                self.fit_R(generator_R, n_steps=self.n_steps_catch_training)
        return self



class PAN_clasifieur_AGNO(_PAN_clasifieur):
    """PAN AGNO"""
    
    name = "PAN_AGNO"
    def sample_z(self, X, width):
        z = np.random.normal( loc=1, scale=width, size=(X.shape[0]) )
        return z

