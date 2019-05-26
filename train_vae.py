# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:29:57 2019

@author: YQ
"""

from seq_vae import SeqVAE
from dataloader import Dataloader

import numpy as np
import tensorflow as tf

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)

vocab_size = 20000+1
train_path = "data/imdb_train.p"
valid_path = "data/imdb_valid.p"
batch_size = 64
n_epochs = 20

train_graph = tf.Graph()
val_graph = tf.Graph()

with train_graph.as_default():
    it, seq = Dataloader(train_path, batch_size).get_iterator()
    m = SeqVAE(vocab_size, training=True)
    m.build(seq)
    
with val_graph.as_default():
    vit, vseq = Dataloader(valid_path, batch_size).get_iterator()
    n = SeqVAE(vocab_size, training=False)
    n.build(vseq)    

sess = tf.Session(graph=train_graph)
val_sess = tf.Session(graph=val_graph)
sess.run(m.init)

writer = tf.summary.FileWriter("vae/", sess.graph)

for i in range(n_epochs):
    sess.run(it.initializer)
    val_sess.run(vit.initializer)
    
    train_loss = []
    val_loss = []
    val_acc = []
    while True:
        try:
            _, loss, step, summ = sess.run([m.op, m.loss, m.step, m.summ_op])
            train_loss.append(loss)
            if step%10 == 0:
                writer.add_summary(summ, step)
            
        except tf.errors.OutOfRangeError:
            m.saver.save(sess, "vae/vae-{}".format(i+1))
            n.saver.restore(val_sess, "vae/vae-{}".format(i+1))
            break
            
    while True:
        try:
            loss, acc = val_sess.run([n.loss, n.accuracy])
            val_loss.append(loss)
            val_acc.append(acc)
        except tf.errors.OutOfRangeError:
            break
    
    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)
    val_acc = np.mean(val_acc)
    print("{}\tTrain loss: {:.4f}  Val loss: {:.4f}  Val acc: {:.2f}".format(i+1, train_loss, val_loss, val_acc))
        
        
sess.close()
val_sess.close()