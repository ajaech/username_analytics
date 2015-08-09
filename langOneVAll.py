import code
import collections
import csv
import Classifier
import gzip
import numpy
import os
import pandas
import langid
import segmenter

from sklearn import metrics
from sklearn.metrics import classification_report
from Classifier import BayesClassifier

numpy.random.seed(666)


def capital_encode(username):
  prev_is_letter = False
  prev_is_capital = False

  out = []
  for char in username:
    if char.isalpha() and prev_is_letter and not prev_is_capital and char.isupper():
      out.append('$')
    out.append(char.lower())
    prev_is_letter = char.isalpha()
    prev_is_capital = char.isupper()

  return ''.join(out)


def load_twitter(filename):
  with gzip.open(filename, 'rU') as f:
    d = pandas.DataFrame([line for line in csv.DictReader(f)])

  d.lang = d.lang.apply(str.strip)
  d.name = d.name.apply(str.strip)
  d.drop_duplicates(cols=['name'], inplace=True)
  d['name_lower'] = d.name.apply(capital_encode)
  #d['name_lower'] = d.name
  return d


if os.path.exists('lang_train_cache.csv'):
  train = pandas.DataFrame.from_csv('lang_train_cache.csv')
  test = pandas.DataFrame.from_csv('lang_test_cache.csv')
else:
  print 'can not load from cache'
  d = load_twitter('data/lang/new_lang_data.txt.gz')
  lang_counts = d.groupby('lang')['lang'].agg('count')
  langs = set(lang_counts[lang_counts.values > 10000].index)
  d = d[d.lang.apply(lambda x: x in langs)]  # use only big languages
  
  langid.set_languages(langs)

  langid_labels = []
  langid_scores = []
  for i, idx in enumerate(d.index):
    if i % 10000 == 0:
      print i
    langid_label, langid_score = langid.classify(d.text[idx])
    langid_labels.append(langid_label)
    langid_scores.append(langid_score)
  d['lid_label'] = langid_labels
  d['lid_score'] = langid_scores

  d = d[(d.lid_score > 0.995) & (d.lang == d.lid_label)]

  # random partioning
  mask = numpy.random.rand(len(d)) < 0.8
  train = d[mask]
  test = d[~mask]
  train.to_csv('lang_train_cache.csv', encoding='utf8')
  test.to_csv('lang_test_cache.csv', encoding='utf8')


def getProbabilities(classifier):
  results = []
  for i in test.index:
    name = test.name_lower[i]
    lang = test.lang[i]

    result = classifier.Classify(name)
    result['lang'] = lang
    result['name'] = name
    results.append(result)
  return pandas.DataFrame(results)


def get_preds(baseline, morph, weight):
  columns = numpy.array(['True', 'False'])
  z = weight * baseline[columns] + (1.0 - weight) * morph[columns]
  idx = z.values.argmax(axis=1)
  return columns[idx]

base_segmenter = segmenter.baseline_segmenter
morph_segmenter = segmenter.morph_segmenter(Classifier.model)

def getMetrics(truelabels, predlabels):
  prec = metrics.precision_score(truelabels, predlabels, pos_label='True')
  recall = metrics.recall_score(truelabels, predlabels, pos_label='True')
  return prec, recall


all_langs = train.lang.unique()
for lang in all_langs:
  labels = [str(x) for x in train.lang == lang]
  testlabels = [str(x) for x in test.lang == lang]
  baseline_classifier = BayesClassifier.Train(base_segmenter,
                                              train.name_lower,
                                              labels)
  morph_classifier = BayesClassifier.Train(morph_segmenter,
                                           train.name_lower,
                                           labels)

  baseline_results = getProbabilities(baseline_classifier)
  morph_results = getProbabilities(morph_classifier)

  preds_morph = get_preds(baseline_results, morph_results, 0.0)
  preds_baseline = get_preds(baseline_results, morph_results, 1.0)
  preds_combo = get_preds(baseline_results, morph_results, 0.5)

  print 'language {0}'.format(lang)
  prec, recall = getMetrics(testlabels, preds_morph)
  print 'morph prec {0} recall {1}'.format(prec, recall)
  prec, recall = getMetrics(testlabels, preds_combo)
  print 'combo prec {0} recall {1}'.format(prec, recall)
  prec, recall = getMetrics(testlabels, preds_baseline)
  print 'baseline prec {0} recall {1}'.format(prec, recall)


def writeConfusion(outfile, counts):
  total = sum(counts.values())
  correct = [counts[p] for p in counts if p[0] == p[1]]
  
  outfile.write('pred,lang,count\n')
  for pred,lang in counts:
    outfile.write('{0},{1},{2}\n'.format(pred, lang, counts[(pred,lang)]))

acc = (preds_morph == morph_results.lang).sum() / float(len(preds_morph))
print 'accuracy morph {0}'.format(acc)

acc = (preds_combo == morph_results.lang).sum() / float(len(preds_combo))
print 'accuracy {0}'.format(acc)
counts = collections.Counter(zip(preds_combo, morph_results.lang))
with open('morph_confusion.csv', 'w') as f:
  writeConfusion(f, counts)

acc = (preds_baseline == morph_results.lang).sum() / float(len(preds_baseline))
print 'baseline accuracy {0}'.format(acc)
counts = collections.Counter(zip(preds_baseline, morph_results.lang))
with open('baseline_confusion.csv', 'w') as f:
  writeConfusion(f, counts)
