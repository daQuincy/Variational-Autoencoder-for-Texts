#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:31:59 2019

@author: yq
"""

import tensorflow as tf
import numpy as np

class SeqVAE:
    def __init__(self, n_vocab, embedding_path=None,
                 rnn_units=512, embedding_size=300, latent_size=128,
                 anneal_steps=5000, beam_width=3,
                 training=False):
        self.n_vocab = n_vocab
        
        self.rnn_units = rnn_units
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.training = training
        
        self.anneal_steps = anneal_steps
        self.beam_width = beam_width
        
        if embedding_path is None:
            self.embeddings = tf.Variable(tf.random_uniform([n_vocab, embedding_size], -1.0, 1.0), name="embeddings")
        else:
            embedding_matrix = np.load(embedding_path)
            self.embeddings = tf.get_variable(shape=embedding_matrix.shape,
                                              initializer=tf.constant_initializer(embedding_matrix),
                                              trainable=False,
                                              name="embedings")
            
        if self.training:
            self.dropout = 0.6
            self.word_dropout = 0.8
        else:
            self.dropout = 1.0
            self.word_dropout = 1.0
        
    def encoder(self, x, seq_len):
        with tf.variable_scope("encoder"):
            rnn = tf.nn.rnn_cell.BasicLSTMCell
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn(self.rnn_units), 
                                                    state_keep_prob=self.dropout, input_keep_prob=self.dropout)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn(self.rnn_units), 
                                                    state_keep_prob=self.dropout, input_keep_prob=self.dropout)
            
            x = tf.nn.embedding_lookup(self.embedding, x)
            
            out, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, x, sequence_length=seq_len-1, dtype=tf.float32)
            final_state = tf.concat([state_fw.h, state_bw.h], -1)
            
            
            z_mean = tf.layers.dense(final_state, self.latent_size, name="mean")
            z_sigma = tf.layers.dense(final_state, self.latent_size, activation=tf.nn.softplus, name="sigma")
            
            return z_mean, z_sigma
        
    def decoder(self, z, seq_len, y):
        with tf.variable_scope("decoder"):
            c = tf.layers.dense(z, self.rnn_units, activation=tf.nn.tanh)
            h = tf.layers.dense(z, self.rnn_units, activation=tf.nn.tanh)
            init_state = tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)
            dense = tf.layers.Dense(self.n_vocab)
            
            rnn = tf.nn.rnn_cell.BasicLSTMCell
            
            cell = rnn(self.rnn_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout)
            
            start_inputs = tf.zeros((tf.shape(z)[0],), dtype=tf.int32)
            
            y = tf.concat([tf.zeros((tf.shape(z)[0], 1), tf.int32), y[:, :-1]], axis=1)
            y = tf.nn.embedding_lookup(self.embedding, y)
            train_out, train_state = tf.nn.dynamic_rnn(cell, y, sequence_length=seq_len+1, initial_state=init_state)
            
            logits = dense(train_out)
            train_output = tf.argmax(logits, axis=-1, output_type=tf.int32)
            
            
            init_state = tf.contrib.seq2seq.tile_batch(init_state, self.beam_width)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell, 
                                                           embedding=self.embedding, 
                                                           start_tokens=start_inputs, 
                                                           end_token=0,
                                                           initial_state=init_state, 
                                                           beam_width=self.beam_width,
                                                           output_layer=dense)
            
            infer_output, _, infer_len = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=50)
            infer_output = infer_output.predicted_ids
            infer_output = tf.transpose(infer_output, [0, 2, 1])
    
        return train_output, infer_output, infer_len, logits
    
    def build(self, X=None):
        if X is None:
            self.X = tf.placeholder(tf.int32, (None, None))
        else:
            self.X = X
            
        X_len = tf.reduce_sum(tf.sign(self.X), axis=-1)
        self.step = tf.train.create_global_step()
               
        # word dropout
        if self.training:
            self.embedding = tf.nn.dropout(self.embeddings, keep_prob=self.word_dropout, noise_shape=[self.n_vocab, 1])
        else:
            self.embedding = self.embeddings
        
        z_mean, z_sigma = self.encoder(self.X, X_len)
        eps = tf.random_normal(tf.shape(z_mean), 0.0, 1.0, tf.float32)
        self.z = z_mean + tf.multiply(z_sigma, eps)
        
        labels = self.X
        train_output, self.output, self.len, logits = self.decoder(self.z, X_len, labels)
        
        mask = tf.stop_gradient(tf.sequence_mask(X_len+1,  dtype=tf.float32))
        
        labels = tf.one_hot(labels, self.n_vocab)
        reconstruction_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        reconstruction_loss = tf.losses.compute_weighted_loss(reconstruction_loss, weights=mask)
        
        k = self.anneal_steps
        beta = tf.minimum(tf.cast(1.0 - k / (k + tf.exp(self.step / k)), tf.float32), 1.0)
        kl_loss_ = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_sigma) - tf.log(tf.square(z_sigma)+1e-8) - 1, axis=-1)
        self.kl_loss_ = tf.reduce_mean(kl_loss_ / self.latent_size)
        kl_loss = beta * self.kl_loss_
        
        self.loss = reconstruction_loss + kl_loss
        self.accuracy = tf.contrib.metrics.accuracy(tf.argmax(labels, -1, output_type=tf.int32), train_output, weights=mask)
        
        learning_rate = tf.train.exponential_decay(1e-3, self.step, 5000, 0.96)
        opt = tf.train.AdamOptimizer(learning_rate)
        g, v = zip(*opt.compute_gradients(self.loss, tf.trainable_variables()))
        g, _ = tf.clip_by_global_norm(g, 5.0)
        gvs = zip(g, v)
        self.op = opt.apply_gradients(gvs, global_step=self.step)
        
        self.init = tf.global_variables_initializer()
        
        summaries = []
        with tf.variable_scope("metrics"):
            summaries += [tf.summary.scalar("reconstruction_loss", reconstruction_loss),
                          tf.summary.scalar("kl_loss", self.kl_loss_),
                          tf.summary.scalar("loss", self.loss),
                          tf.summary.scalar("accuracy", self.accuracy)]
        with tf.variable_scope("misc"):
            summaries += [tf.summary.scalar("beta", beta)]
                          #tf.summary.scalar("learning_rate", learning_rate)]
        
        self.summ_op = tf.summary.merge(summaries)
        self.saver = tf.train.Saver()


if __name__ == "__main__":
    # for debugging
    tf.reset_default_graph()
    m = SeqVAE(128, training=False)
    m.build()