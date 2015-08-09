import code
import morfessor
import os
import pandas

import MyClassifier


data = pandas.read_csv('data/cupid/okcupid_usernames.csv', skipinitialspace=True)
guy_names = data.name[data.gender == 'guy']
girl_names = data.name[data.gender == 'girl']


model_path = 'snapchat/model0.7.bin'

io = morfessor.MorfessorIO()

MyClassifier.model = io.read_binary_model_file(model_path)

print 'doing experiment for baseline segmenter'
MyClassifier.DoTest(guy_names, girl_names, unsupervised=False,
                    balance=True, use_baseline_segmenter=True)

print 'doing experiment for u-morph segmenter'
MyClassifier.DoTest(guy_names, girl_names, unsupervised=False,
                    balance=True, use_baseline_segmenter=False)
