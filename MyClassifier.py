import code
import collections
import gzip
import math
import morfessor
import numpy
import pandas
import random
import segmenter
import sklearn.metrics

from matplotlib import pylab

random.seed(666)
model = None

def LoadUsernames(filename, maxload=400000000):
  if filename.endswith('.gz'):
    f = gzip.open(filename, 'r')
  else:
    f = open(filename, 'r')
  usernames = []
  for i, line in enumerate(f):
    if i > maxload:
      break
    usernames.append(line.strip())
  f.close()
  random.shuffle(usernames)
  return usernames


class BinaryClassifier:

  alpha = 2.0  # smoothing parameter

  def __init__(self, segfun, guy_morphs, girl_morphs,
               guy_count, girl_count):
    self.segfun = segfun
    
    self.guy_morphs = guy_morphs
    self.girl_morphs = girl_morphs

    self.guy_count = guy_count
    self.girl_count = girl_count

    self.confidences = None
    self.confidence_bins = None


  @classmethod
  def Train(cls, segfun, guy_train, girl_train):

    guy_morphs = collections.defaultdict(int)
    girl_morphs = collections.defaultdict(int)

    tempfile = open('segments.txt', 'w')

    for name in guy_train:
      segments = segfun(name.lower())
      msg = u'male: {0} -- {1}\n'.format(name.decode('utf8'), u'*'.join(segments))
      tempfile.write(msg.encode('utf8'))
      for seg in segments:
        guy_morphs[seg] += 1
    for name in girl_train:
      segments = segfun(name.lower())
      msg = u'female: {0} -- {1}\n'.format(name.decode('utf8'), u'*'.join(segments))
      tempfile.write(msg.encode('utf8'))
      for seg in segments:
        girl_morphs[seg] += 1

    tempfile.close()

    return cls(segfun, guy_morphs, girl_morphs, len(guy_train),
               len(girl_train))


  @classmethod
  def TrainSemiSupervised(cls, usernames, classifier, unlabeled_weight = 0.3):
    segfun = classifier.segfun

    guy_morphs = collections.defaultdict(int, classifier.guy_morphs)
    girl_morphs = collections.defaultdict(int, classifier.girl_morphs)
    
    guy_count = classifier.guy_count
    girl_count = classifier.girl_count

    for name in usernames:
      score, segments = classifier.Classify(name.lower(), return_segments=True)
      guy_prob = classifier.GetConfidence(score)
      girl_prob = 1.0 - guy_prob

      # skip people that the classifier is uncertain about
      if abs(guy_prob - 0.5) < 0.1:
        continue

      guy_prob *= unlabeled_weight
      girl_prob *= unlabeled_weight
      
      guy_count += guy_prob
      girl_count += girl_prob

      for seg in segments:
        guy_morphs[seg] += guy_prob
        girl_morphs[seg] += girl_prob

    return cls(segfun, guy_morphs, girl_morphs, guy_count,
               girl_count)


  def GetTopRatios(self):
    stats = []

    all_morphs = set(self.guy_morphs.keys() + self.girl_morphs.keys())
    print 'vocabulary size {0}'.format(len(all_morphs))

    # compute the average morph length     
    avg_morph_len = sum([len(morph) for morph in all_morphs]) / float(len(all_morphs))
    print 'average morph length {0}'.format(avg_morph_len)

    morphs = [self.guy_morphs, self.girl_morphs]
    class_totals = [sum(morphs[i].values()) + self.alpha * len(all_morphs) 
                    for i in range(len(morphs))]
    class_totals = [float(c) for c in class_totals]

    for token in all_morphs:
      guy_prob = (morphs[0][token] + self.alpha)  / class_totals[0]
      girl_prob = (morphs[1][token] + self.alpha) / class_totals[1]
      
      ratio = numpy.log(guy_prob) - numpy.log(girl_prob)
      stats.append({'morph': token, 'ratio': ratio, 'weight': numpy.abs(ratio),
                    'guy count': self.guy_morphs[token],
                    'girl count': self.girl_morphs[token]})

    d = pandas.DataFrame(stats)
    d.sort('weight', inplace=True, ascending=False)
    print d[:20]

  
  def TrainConfidenceEstimator(self, guy_names, girl_names):
    guy_scores = [self.Classify(name) for name in guy_names]
    girl_scores = [self.Classify(name) for name in girl_names]

    total = sorted(guy_scores + girl_scores)
    bins = numpy.percentile(total, range(0, 101, 10))

    guy_bin_counts, _ = numpy.histogram(guy_scores, bins)
    girl_bin_counts, _ = numpy.histogram(girl_scores, bins)

    confidences = numpy.array(guy_bin_counts, dtype=float) / (
      guy_bin_counts + girl_bin_counts)
    
    self.confidences = confidences
    self.confidence_bins = bins


  def GetConfidence(self, score):
    bin = numpy.histogram([score], self.confidence_bins)[0]
    confidence = self.confidences[numpy.argmax(bin)]

    return confidence


  def Classify(self, username, return_segments=False, alpha=None,
               oov_counter=None):
    if alpha is None:
      alpha = self.alpha

    segments = self.segfun(username.lower())

    p_of_c = float(self.guy_count) / float(self.girl_count + self.guy_count)

    guy_prob = math.log(p_of_c)
    girl_prob = math.log(1.0 - p_of_c)

    guy_denom = 1.0 / (self.guy_count + 2 * alpha)
    girl_denom = 1.0 / (self.girl_count + 2 * alpha)

    for seg in segments:
      guy_count = self.guy_morphs.get(seg, 0)
      girl_count = self.girl_morphs.get(seg, 0)

      if oov_counter is not None:
        if guy_count + girl_count == 0:
          oov_counter[seg] += 1

      guy_prob += math.log((guy_count + 2.0 * p_of_c * alpha) * guy_denom)
      girl_prob += math.log((girl_count + 2.0 * (1.0 - p_of_c) * alpha) * girl_denom)

    score = guy_prob - girl_prob

    if return_segments:
      return score, segments

    return score


def Partition(data, percents):
  cutoffs = [int(math.floor(len(data) * p)) for p in numpy.cumsum(percents)]
  return data[:cutoffs[0]], data[cutoffs[0]:cutoffs[1]], data[cutoffs[1]:]


def GetRocCurve(a_scores, b_scores):
  all_scores = a_scores + b_scores
  labels = [1 for _ in a_scores] + [-1 for _ in b_scores]

  fpr, tpr, thresh = sklearn.metrics.roc_curve(labels, all_scores)

  return fpr, tpr, thresh


def TestAccuracy(classifier, classA, classB, threshold):
  oov_counts = collections.defaultdict(int)
  all_morphs = set(classifier.guy_morphs.keys() + classifier.girl_morphs.keys())
  print 'vocabulary size {0}'.format(len(all_morphs))

  a_scores = [classifier.Classify(name, oov_counter=oov_counts) for name in classA]
  b_scores = [classifier.Classify(name, oov_counter=oov_counts) for name in classB]

  print 'total # of oovs types: {0} tokens: {1}'.format(
    len(oov_counts), sum(oov_counts.values()))

  num_correct = (b_scores < threshold).sum() + (a_scores >= threshold).sum()
  acc = num_correct / float(len(classA) + len(classB))
  return acc


def GetOptimalThreshold(classifier, classA, classB):
  smooth_levels = (1.0, 2.0, 5.0, 7.0, 9.0)
  results = []
  for alpha in smooth_levels:
    a_scores = [classifier.Classify(name, alpha=alpha) for name in classA]
    b_scores = [classifier.Classify(name, alpha=alpha) for name in classB]
  
    fpr, tpr, thresh = GetRocCurve(a_scores, b_scores)

    idx = (1.0 - fpr) < tpr
    crossover = numpy.where(idx)[0].min()
    acc = 0.5 * (tpr[crossover] + tpr[crossover-1])

    auc = sklearn.metrics.auc(fpr, tpr)

    results.append({'thresh': thresh[crossover], 'accuracy': acc,
                    'smooth': alpha})
  
  data = pandas.DataFrame(results)
  idx = numpy.argmax(data.accuracy)

  classifier.alpha = data.smooth[idx]
  return data.thresh[idx]


num_semisup = 2000000
snapchat_names = LoadUsernames('snapchat/test_usernames.txt.gz',
                               maxload=num_semisup)
print '{0} semisup names loaded'.format(len(snapchat_names))
  

def DoTest(classA, classB, unsupervised=False, balance=False,
           use_baseline_segmenter=False):
  classA = list(classA)
  classB = list(classB)

  if balance:
    max_len = min(len(classA), len(classB))
    classA = classA[:max_len]
    classB = classB[:max_len]

  random.shuffle(classA)
  random.shuffle(classB)
  percents = (0.2, 0.1, 0.7)
  classA_test, classA_validation, classA_train = Partition(classA, percents)
  classB_test, classB_validation, classB_train = Partition(classB, percents)

  if use_baseline_segmenter:
    seg_func = segmenter.baseline_segmenter
  else:
    seg_func = segmenter.morph_segmenter(model)

  classifier = BinaryClassifier.Train(seg_func, classA_train, classB_train)
  classifier.GetTopRatios()

  thresh = GetOptimalThreshold(classifier,
                               classA_validation, classB_validation)
  
  acc = TestAccuracy(classifier, classA_test, classB_test, thresh)
  print 'test accuracy {0}'.format(acc)

  if unsupervised:
    semisup_classifier = classifier
    for iter_num in range(3):
      print 'semi-sup iter {0}'.format(iter_num)
      semisup_classifier.TrainConfidenceEstimator(classA_validation, classB_validation)

      semisup_classifier = BinaryClassifier.TrainSemiSupervised(snapchat_names,
                                                                semisup_classifier)
      thresh = GetOptimalThreshold(semisup_classifier, classA_validation, classB_validation)
      acc = TestAccuracy(semisup_classifier, classA_test, classB_test, thresh)
      print 'accuracy {0}'.format(acc)
