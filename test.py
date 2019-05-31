# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:47:29 2019

@author: YQ
"""

from seq_vae import SeqVAE
import pickle
import tensorflow as tf
import re
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", default="generate", type=str)
ap.add_argument("-s", "--sentence", type=str)
ap.add_argument("-s1", "--sentence_1", type=str)
ap.add_argument("-s2", "--sentence_2", type=str)
ap.add_argument("-b", "--beam_size", default=1, type=int)
ap.add_argument("-r", "--restore_path", default="vae/vae-25", type=str)
args = ap.parse_args()

def preprocess_sentence(s):
    s = s.lower()
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿']+", " ", s)
    s = s.rstrip().strip()
    
    s = tokenizer.texts_to_sequences([s])
    s = np.array(s).reshape(1, -1).astype(np.int32)
    
    return s

def slerp(p0, p1, t):
    """Spherical interpolation."""
    p0 = p0[0]
    p1 = p1[0]
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    ss = np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1
      
    return np.expand_dims(ss, 0)

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
tokenizer = pickle.load(open("data/imdb_tokenizer.p", "rb"))
vocab_size = 20000 + 1

def load_model():
    m = SeqVAE(vocab_size, beam_width=args.beam_size, training=False)
    m.build()
    
    sess = tf.Session()
    m.saver.restore(sess, args.restore_path)
    
    return m, sess

def interpolate(m, sess):
    seq1 = preprocess_sentence(args.sentence_1)
    seq2 = preprocess_sentence(args.sentence_2)
    
    z1 = sess.run(m.z, feed_dict={m.X: seq1})
    z2 = sess.run(m.z, feed_dict={m.X: seq2})
                                  
    interpolations = []
    g_tmp, length = sess.run([m.output, m.len], feed_dict={m.z: z1})
    interpolations.append(g_tmp[0, 0][:length[0, 0]-1])
    for i in np.linspace(0, 1, 10):
        z_tmp = slerp(z1, z2, i)
        g_tmp, length = sess.run([m.output, m.len], feed_dict={m.z: z_tmp})
        interpolations.append(g_tmp[0, 0][:length[0, 0]-1])
        
    g_tmp, length = sess.run([m.output, m.len], feed_dict={m.z: z2})
    interpolations.append(g_tmp[0, 0][:length[0, 0]-1])
    
    for o in interpolations:
        tmp = o.tolist()
        tmp = tokenizer.sequences_to_texts([tmp])
        print(" ".join(tmp))
        print(" ")
        
def generate(m, sess):
    z = np.random.normal(size=(1, 128))
    vector, length = sess.run([m.output, m.len], feed_dict={m.z: z})
    
    for v, l in zip(vector[0], length[0]):
        tmp = v[:l-1].tolist()
        tmp = tokenizer.sequences_to_texts([tmp])
        print(" ".join(tmp))
        
def reconstruct(m, sess):
    sentence = preprocess_sentence(args.sentence)
    vector, length = sess.run([m.output, m.len], feed_dict={m.X: sentence})
    for v, l in zip(vector[0], length[0]):
        tmp = v[:l-1].tolist()
        tmp = tokenizer.sequences_to_texts([tmp])
        print(" ".join(tmp))
        
if __name__ == "__main__":
    tf.reset_default_graph()
    m, sess = load_model()
    if args.mode == "interpolate":
        interpolate(m, sess)
    elif args.mode == "generate":
        generate(m, sess)
    elif args.mode == "reconstruct":
        reconstruct(m, sess)
        
    sess.close()
