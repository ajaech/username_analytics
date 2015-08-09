import code
import collections
import gzip
import math
import numpy
import pandas
import random
import morfessor


io = morfessor.MorfessorIO()
model = io.read_binary_model_file('data/lang/model0.7.bin')

class BayesClassifier:

  alpha = 400000.0  # smoothing parameter

  def __init__(self, segfun, morphs, class_counts):

    self.segfun = segfun
    
    self.morphs = morphs
    self.counts = class_counts  # for calculating class membership probabilities
    print class_counts

    # for calculating within class token probabilities
    self.lengths = dict([(x, sum(morphs[x].values())) for x in morphs])

    classes = [morphs[x].keys() for x in morphs]
    all_morphs = set()
    for i in classes:
      all_morphs.update(i)
    self.vocab_size = len(all_morphs)

    self.total_count = float(sum(class_counts.values()))


  @classmethod
  def Train(cls, segfun, usernames, labels):
    morphs = collections.defaultdict(collections.Counter)
    class_counts = collections.Counter(labels)

    for i, (name, label) in enumerate(zip(usernames, labels)):
      if i % 10000 == 0:
        print "progress {0} of {1}".format(i, len(labels))

      morphs[label].update(segfun(name))

    return cls(segfun, morphs, class_counts)

  
  def Classify(self, username, oov_counter=None):
    segments = self.segfun(username)
    classes = self.counts.keys()
    num_classes = len(classes)

    # Z normalizes the probability distribution, not really necessary
    Z = 1.0 / (self.vocab_size + self.alpha)

    class_logpriors = dict([(c, math.log(self.counts[c] / self.total_count))
                            for c in self.counts.keys()])

    classes = self.counts.keys()

    # keep track of oov u-morphs
    if oov_counter is not None:
      for seg in segments:
        total_count = sum([self.morphs[c][seg] for c in classes])
        if total_count == 0:
          oov_counter[seg] += 1

    log_likelihoods = []
    for c in classes:
      likelihood = class_logpriors[c]

      # class dependent scaling
      scale = self.alpha / self.lengths[c]

      morph_counts = self.morphs[c]
      for seg in segments:
        p_w_given_c = 1.0 + scale * morph_counts[seg]
        likelihood += math.log(Z * p_w_given_c)
      log_likelihoods.append(likelihood)

    return dict(zip(classes, log_likelihoods))
