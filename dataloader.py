# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:05:19 2019

@author: YQ
"""

import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataloader:
    def __init__(self, path, batch_size=64):
        self.data = pickle.load(open(path, "rb"))
        self.batch_size = batch_size
        self.total_batches = int(len(self.data) // self.batch_size)
        
    def loader(self):       
        for i in range(self.total_batches):
            tmp = self.data[self.batch_size*i:(self.batch_size*i)+self.batch_size]
            if len(tmp) == self.batch_size:
                yield pad_sequences(tmp, padding="post")
            else:
                break
            
    def get_iterator(self):
        ds = tf.data.Dataset.from_generator(
                self.loader, (tf.int32))
        ds = ds.shuffle(128)
        
        iterate = ds.make_initializable_iterator()
        seq = iterate.get_next()
        seq.set_shape([self.batch_size, None])
        
        return iterate, seq
    

if __name__ == "__main__":
    it, seq = Dataloader("sentiment_valid.p").get_iterator()
    
    sess = tf.Session()
    sess.run(it.initializer)
    
    data = []
    while True:
        try:
            data.append(sess.run(seq))
        except:
            break