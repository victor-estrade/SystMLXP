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

from ._nn import TFClassifier


class NN(TFClassifier):
    """Simple Neural net"""
    def __init__(self, input_shape, learning_rate=0.01, optimizer="Adam", n_iter=None,
                 n_layers=3, n_neurons_per_layer=40, summary_dir=None):
        super(NN, self).__init__()
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_layers = n_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_iter = n_iter
        self.build()
        self.scaler = StandardScaler()
        self._i_run = 0
        if summary_dir is None:
            self.summary_dir = tempfile.mkdtemp()
        else:
            self.summary_dir = os.path.join( summary_dir, self.get_name() )

    def get_name(self):
        name = '{}_{}x{}-{}-{}-{}'.format('NN', self.n_neurons_per_layer, self.n_layers, self.learning_rate, self.optimizer, self.n_iter)
        return name

    def get_summary_dir(self):
        dir_path =  os.path.join(self.summary_dir, 'run-{}'.format(self._i_run))
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        return dir_path

    def build(self):
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            self.in_X = tf.placeholder(tf.float32, shape=(None,)+self.input_shape, name='in_X')
            self.in_y = tf.placeholder(tf.int32, shape=(None,), name='in_y')
            self.in_w = tf.placeholder(tf.float32, shape=(None,), name='in_w')
            y = tf.one_hot(self.in_y, 2, 1.0, 0.0)
            
            previous_layer = self.in_X
            for i in range(self.n_layers):
                dense_i = tf.contrib.layers.fully_connected(previous_layer, self.n_neurons_per_layer, activation_fn=None, scope='dense_{}'.format(i))
                act_i = tf.nn.softplus(dense_i)
                previous_layer = act_i

            dense_final = tf.contrib.layers.fully_connected(previous_layer, 2, activation_fn=None, scope='dense_final')
            output_layer = tf.nn.softmax(dense_final)

            # Calculate losses
            loss = tf.losses.softmax_cross_entropy(y, output_layer, weights=self.in_w)

            # Summaries
            tf.summary.scalar('Classification_Loss', loss)

            # Define operators
            self.op_predict_proba = output_layer
            self.op_predict = tf.argmax(self.op_predict_proba, axis=1)
            self.op_loss = loss
            self.op_train = tf.contrib.layers.optimize_loss(
                loss=self.op_loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.optimizer)
            self.op_init = tf.global_variables_initializer()
        self.init()

    def fit(self, X, y, sample_weight=None, batch_size=512, n_steps=1):
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        self.reset()
        self._i_run += 1
        if self.n_iter is not None:
            n_steps = self.n_iter
        X = self.scaler.fit_transform(X)
        X = np.asarray(X)
        y = np.asarray(y)
        w = np.asarray(sample_weight)
        batch_gen = self.minibatch_generator(X, y, w, batch_size=batch_size)
        
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            summaries = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.get_summary_dir(), sess.graph)
            for i in range(n_steps):
                X_batch, y_batch, w_batch = next(batch_gen)
                feed_dict = {self.in_X: X_batch, 
                               self.in_y:y_batch,
                               self.in_w: w_batch}
                loss = sess.run(self.op_train,
                                feed_dict=feed_dict)
                if ( i % 20 ) == 0 :
                    summary_str = sess.run(summaries, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            prediction = sess.run(self.op_predict, feed_dict={self.in_X: X})
        return prediction

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            proba = sess.run(self.op_predict_proba, feed_dict={self.in_X: X})
        return proba

class NNA(NN):
    def __init__(self, input_shape, learning_rate=0.01, optimizer="Adam", n_iter=None, width=0.01,
                 n_layers=3, n_neurons_per_layer=40, summary_dir=None, skew=None):
        super(NN, self).__init__()
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_layers = n_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.n_iter = n_iter
        self.skew = skew
        self.width = width
        self.build()
        self.scaler = StandardScaler()
        self._i_run = 0
        if summary_dir is None:
            self.summary_dir = tempfile.mkdtemp()
        else:
            self.summary_dir = os.path.join( summary_dir, self.get_name() )
    
    def get_name(self):
        name = '{}_{}x{}-{}-{}-{}-{}'.format('NNA', self.n_neurons_per_layer, self.n_layers, self.learning_rate, 
        	self.optimizer, self.n_iter, self.width)
        return name

    def sample_z(self, X):
        z = np.round( np.random.normal( loc=1, scale=self.width, size=(X.shape[0]) ), 2 )
        return z

    def fit(self, X, y, batch_size=512, n_steps=1):
        self.reset()
        self._i_run += 1
        if self.n_iter is not None:
            n_steps = self.n_iter
        z = self.sample_z(X)
        X = self.skew(X, z)
        X = self.scaler.fit_transform(X)
        X = np.asarray(X)
        y = np.asarray(y)
        batch_gen = self.minibatch_generator(X, y, batch_size=batch_size)
        
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            summaries = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.get_summary_dir(), sess.graph)
            for i in range(n_steps):
                X_batch, y_batch = next(batch_gen)
                feed_dict = {self.in_X: X_batch, 
                               self.in_y:y_batch}
                loss = sess.run(self.op_train,
                                feed_dict=feed_dict)
                if ( i % 20 ) == 0 :
                    summary_str = sess.run(summaries, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()
        return self

