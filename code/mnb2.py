from math import log, sqrt
from collections import defaultdict
from util import *
import random
from nltk.stem import PorterStemmer
import re
import itertools

stop = set([ l.strip() for l in open('english.stop')])

def strip(doc):
  ps = PorterStemmer()
  words = re.findall('[\w-]+', re.sub('\$.*?\$','math123', doc.lower()))
  return [ps.stem(w) for w in words if w not in stop and len(w) >= 4]

def _vocab(corpus):
  words = set([])
  df = Counter()
  for d in corpus:
    wrds = strip(d)
    for w in set(wrds):
      df[w] += 1
    for t in wrds:
      # if df[t] > 5:
        words.add(t)
  return (words, df)

def word_counts(corpus, ids):
  n = {}
  for d in zip(corpus, ids):
    n[d[1]] = defaultdict(lambda: 0)
    for w in strip(d[0]):
      n[d[1]][w] += 1
  return n

def tfidf(w, f, df, d):
  return log(f[w]/float(len(f)) + 1.0) * log( float(d) / df[w])

def normalize_tfidf(n, df, d):
  np = {}
  for doc in n.keys():
    np[doc] = defaultdict(lambda:0)
    for w in n[doc].keys():
      t = tfidf(w, n[doc], df, d)
      if t > 0:
        np[doc][w] = t
  return np

def normalize_len(n):
  np = {}
  avg = 0
  for doc in n.keys():
    avg += sqrt(sum([n[doc][x]**2 for x in n[doc].keys()]))
  avg /= len(n.keys())
  print avg
  for doc in n.keys():
    np[doc] = defaultdict(lambda:0)
    for w in n[doc].keys():
      if n[doc][w] > 0:
        np[doc][w] = n[doc][w] / avg
  return np

def normalize_class_length(classes, labels, vocab, n):
  np = {}
  for c in classes:
    tot = 0
    np[c] = {}
    for d in n.keys():
      if labels[d] == c:
        for w in n[d].keys():
          tot += n[d][w]
    for d in n.keys():
      np[c][d] = defaultdict(lambda:0)
      for w in n[d].keys():
        np[c][d][w] = float(1000*n[d][w]) / tot
  return np

def train(classes, vocab, corpus, corpus_labels, feats, perClass=False):
  print 'Extracting vocab'
  # vocab = _vocab(corpus[0])[0]
  d = len(corpus[0])
  probs = {} #probs[c][w] = P(w|c)
  prior = {}
  for c in classes:
    n = feats
    if perClass:
      n = feats[c]
    print 'Training %s' % c
    doc_class_ids = [x[1] for x in zip(corpus_labels, corpus[1]) if x[0] == c]
    probs[c] = defaultdict(lambda:1.0 / len(vocab))
    prior[c] = len(doc_class_ids) / float(d)
    totalCount = 0
    sum_words = defaultdict(lambda: 0)
    
    for idx in doc_class_ids:
      for w in n[idx].keys():
        totalCount += n[idx][w]
        sum_words[w] += n[idx][w]

    for w in vocab:
      top = sum_words[w]
      if top > 0:
        probs[c][w] = (top + 1) / float(totalCount -top + d)
  return vocab, prior, probs

def label(classes, vocab, prior, probs, doc):
  words = [t for t in strip(doc) if t in vocab]
  score = {}
  for c in classes:
    score[c] = log(prior[c])
    for t in words:
      score[c] += log(probs[c][t])

  return max(score, key=score.get), score

def main(): 
  classes = ['physics', 'cs', 'math', 'stat']

  total = load_training()
  random.shuffle(total)
  labels = load_label()
  frac_train = int(len(total)*1.0)
  print frac_train
  training = total[:frac_train]
  test = total[frac_train:]

  print 'Extracting Corpus'
  corpus = [d['abs'] for d in training]
  ids = [d['id'] for d in training]
  corpus_labels = [labels[d['id']] for d in training]
  # print 'Training'
  # n = word_counts(corpus, ids)
  # print 'Temp Vocab'
  # vocab, df = _vocab(corpus)
  # print 'Calculating TF-IDF'
  # np = normalize_tfidf(n,df,len(corpus))
  # # np2 = normalize_class_length(classes, labels, vocab, n)
  # np2 = normalize_len(np)
  # c1 = train(classes, vocab, (corpus, ids), corpus_labels, np2)
  # measure(c1, classes, test, labels)
  gen = kfold(10, classes, (corpus, ids, corpus_labels))
  for t_data, t_ids, t_labs, test in gen:
    vocab, df = _vocab(corpus)
    d = len(corpus)
    n = word_counts(t_data, t_ids)
    # n = normalize_tfidf(n,df, d)
    # n = normalize_len(n)
    c = train(classes, vocab, (t_data, t_ids), t_labs, feats=n)
    labels = []
    for doc in test:
      labels.append(label(classes, *c, doc=doc)[0])
    print gen.send(labels)
  
def kfold(k, classes, data):
  # data is tuple (text, id, label)
  # params is list of values to test
  k_size = int(round(len(data[0])/ k))
  groups = [(data[0][i:i + k_size],data[1][i:i + k_size],data[2][i:i + k_size]) for i in range(0, len(data[0]), k_size)]
  errors = [0] * k
  for i in range(0,k):
    print 'Fold %d' % i
    test = groups[i]
    t_data, t_ids, t_labs = [],[],[]
    for j in range(0,k):
      if j != i:
        t_data += (groups[j][0])
        t_ids += (groups[j][1])
        t_labs += (groups[j][2])

    preds = yield (t_data, t_ids, t_labs, test[0])
    yield None
    # for d in zip(*test):
    for j in range(len(zip(*test))):
      if preds[j] != zip(*test)[j][2]:
        errors[i] += 1
    print errors
  print float(sum(errors))/len(errors)/len(test[0])


def measure(c, classes, test, labels):
  print 'Testing'
  correct = 0
  both_wrong = 0
  one_right  = 0
  errors = defaultdict(lambda:defaultdict(lambda:0))
  test_length = len(test)
  # print len(test)
  for d in test[:test_length]:
    pred,scores = label(classes,*c,doc=d['abs'])
    # print pred
    if pred == labels[d['id']]:
      correct += 1
    errors[labels[d['id']]][pred] += 1
      # print "%s %s" % (d['id'], scores)
  # print "Got %s correct of %s" % (correct, test_length)  
  print errors

def compare(c1,c2, classes, test, labels):
  print 'Testing'
  correct = [0,0]
  both_wrong = 0
  one_right  = 0
  errors = [defaultdict(lambda:0),defaultdict(lambda:0)]
  test_length = len(test)
  for d in test[:test_length]:
    pred = label(classes,*c1,doc=d['abs'])[0]
    if pred == labels[d['id']]:
      correct[0] += 1
    else:
      errors[0][labels[d['id']]] +=1
    pred2 = label(classes, *c2, doc=d['abs'])[0]
    if pred2 == labels[d['id']]:
      correct[1] += 1
    else:
      errors[1][labels[d['id']]] +=1

    if (pred2 == labels[d['id']]) and (pred == labels[d['id']]):
      one_right += 1
    if not (pred2 == labels[d['id']]) and not (pred == labels[d['id']]):
      both_wrong += 1
  print "Got %s, %s correct of %s" % (correct[0],correct[1], test_length) 
  print both_wrong
  print one_right 
  print errors[0]
  print errors[1]

def output(classes, c):
  test = load_test()
  for doc in test:
    print "\"%s\",\"%s\"" % (doc['id'], label(classes, *c, doc=doc['abs'])[0])

if  __name__ =='__main__':main()