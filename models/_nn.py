# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf 

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from .minibatch import epoch_shuffle


class _TFSave():
    """An abstract class to store and save Tensorflow data.
    Provides : a graph, a session linked to the graph, saving and loading.
    """
    def __init__(self):
        super(_TFSave, self).__init__()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
    
    def save(self, file_path, verbose=0):
        """Saves the model variables on disk."""
        with self.graph.as_default() as g:
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, file_path)
            if verbose:
                print("Model saved in file: {}".format(save_path))
    
    def load(self, file_path, verbose=0):
        """Restore the variables values from disk."""
        with self.graph.as_default() as g:
            saver = tf.train.Saver()
            saver.restore(self.sess, file_path)
            if verbose:
                print("Model restored from file: {}".format(file_path))


class _TFInit(_TFSave):
    """Abstract class for tensorflow models.
    [Reminder] Need to implement :
        - op_init
        - build()
    """
    
    op_init = None

    def __init__(self):
        super(_TFInit, self).__init__()

    def build(self):
        """Abstract method. Build the model. Define variables, operators, etc."""
        raise NotImplementedError

    def reset(self):
        """Reset the model. Renews the graph and session and build() again."""
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.build()
            
    def init(self):
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            sess.run(self.op_init)


class TFBaseModel(_TFInit, BaseEstimator):
    """Abstract class for tensorflow models.
    [Reminder] Need to implement :
        - op_init
        - op_predict
        - op_train
        - minibatch_generator
        - in_X
        - in_y
        - build()
    Optional reimplementation :
        - fit() {recommended}
        - predict()
    """
    op_predict = None
    op_train = None
    op_loss = None
    minibatch_generator = None
    in_X = None
    in_y = None
    
    def __init__(self, minibatch_generator=epoch_shuffle):
        super(TFBaseModel, self).__init__()
        self.minibatch_generator = minibatch_generator

    def set_params(self, **params):
        BaseEstimator.set_params(self, **params)
        self.reset()

    def fit(self, X, y, batch_size=512, n_steps=1):
        # FIXME : need reset AND init or just init ?
        self.reset()
        self.init()
        batch_gen = self.minibatch_generator(X, y, batch_size=batch_size)
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            # TODO : How to handle summaries ?
            # summaries = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter(self.model_dir, g)
            for i in range(n_steps):
                X_batch, y_batch = next(batch_gen)
                feed_dict = {self.in_X: X_batch, self.in_y: y_batch}
                loss = sess.run(self.op_train,
                                feed_dict=feed_dict)
                # TODO : How to handle summaries ?
                # FIXME : Change constant "20" -> self.SUMMARIES_N_STEPS ?
                # if i%20 == 0 :
                #     summary_str = sess.run(summaries, feed_dict=feed_dict)
                #     summary_writer.add_summary(summary_str, i)
                #     summary_writer.flush()
        return self

    def partial_fit(self, X, y, batch_size=512, n_steps=1):
        batch_gen = self.minibatch_generator(X, y, batch_size=batch_size)
        with self.graph.as_default() as g, self.sess.as_default() as sess :
            # TODO : How to handle summaries ?
            # summaries = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter(self.model_dir, g)
            for i in range(n_steps):
                X_batch, y_batch = next(batch_gen)
                feed_dict = {self.in_X: X_batch, self.in_y: y_batch}
                loss = sess.run(self.op_train,
                                feed_dict=feed_dict)
                # TODO : How to handle summaries ?
                # FIXME : Change constant "20" -> self.SUMMARIES_N_STEPS ?
                # if i%20 == 0 :
                #     summary_str = sess.run(summaries, feed_dict=feed_dict)
                #     summary_writer.add_summary(summary_str, i)
                #     summary_writer.flush()
        return self

    def predict(self, X):
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            prediction = sess.run(self.op_predict, feed_dict={self.in_X: X})
        return prediction


class TFClassifier(TFBaseModel, ClassifierMixin):
    """Abstract class for tensorflow models for classification.
    [Reminder] Need to implement :
        - op_init
        - op_predict
        - op_train
        - minibatch_generator
        - in_X
        - in_y
        - build()
    Optional reimplementation :
        - fit() {recommended}
        - predict()
        - predict_proba()
    """
    op_predict_proba = None
    
    def __init__(self):
        super(TFClassifier, self).__init__()

    def predict_proba(self, X):
        with self.graph.as_default() as g, self.sess.as_default() as sess:
            proba = sess.run(self.op_predict_proba, feed_dict={self.in_X: X})
        return proba


class TFRegressor(TFBaseModel, RegressorMixin):
    """Abstract class for tensorflow models for regression.
    [Reminder] Need to implement :
        - op_init
        - op_predict
        - op_train
        - minibatch_generator
        - in_X
        - in_y
        - build()
    Optional reimplementation :
        - fit() {recommended}
        - predict()
    """
    def __init__(self):
        super(TFRegressor, self).__init__()
