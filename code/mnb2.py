from math import log
from collections import defaultdict
from util import *
import random
from nltk.stem import PorterStemmer
import re

def strip(doc):
  stop = set([ l.strip() for l in open('english.stop')])
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
    if int(d[1]) % 1000 == 0:
      print d[1]
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

def normalize_class_length(c, labels, vocab, n):
  np = {}
  tot = 0
  for d in n.keys():
    if labels[d] == c:
      for w in n[d].keys():
        tot += n[d][w]
  for d in n.keys():
    for w in n[d].keys():
      np[d][w] = float(n[d][w]) / tot

def train(classes, vocab, corpus, corpus_labels, n):
  print 'Extracting vocab'
  # vocab = _vocab(corpus[0])[0]
  d = len(corpus[0])
  print len(vocab)
  probs = {} #probs[c][w] = P(w|c)
  prior = {}
  for c in classes: 
    print 'Training %s' % c
    doc_class_ids = [x[1] for x in zip(corpus_labels, corpus[1]) if x[0] == c]
    probs[c] = defaultdict(lambda:1.0 / len(vocab))
    prior[c] = len(doc_class_ids) / float(d)
    totalCount = 0
    i = 0
    sum_words = defaultdict(lambda: 0)
    for idx in doc_class_ids:
      i += 1
      if i % 1000 == 0:
        print i
      for w in n[idx].keys():
        totalCount += n[idx][w]
        sum_words[w] += n[idx][w]
    print totalCount

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
  # random.shuffle(total)
  labels = load_label()
  frac_train = int(len(total)*(.95))
  print frac_train
  training = total[:frac_train][:40000]
  test = total[frac_train:]

  print 'Extracting Corpus'
  corpus = [d['abs'] for d in training]
  ids = [d['id'] for d in training]
  corpus_labels = [labels[d['id']] for d in training]
  print 'Training'
  n = word_counts(corpus, ids)
  print 'Temp Vocab'
  vocab, df = _vocab(corpus)
  print 'Calculating TF-IDF'
  np = normalize_tfidf(n,df,len(corpus))
  c1 = train(classes, vocab, (corpus, ids), corpus_labels, np)
  # c2 = train(classes, vocab, (corpus, ids), corpus_labels, n)
  # compare(c1, c2, classes, test, labels)
  measure(c1, classes, test, labels)
  # output(classes, c1)
def measure(c, classes, test, labels):
  print 'Testing'
  correct = 0
  both_wrong = 0
  one_right  = 0
  errors = defaultdict(lambda:0)
  test_length = 1000
  for d in test[:test_length]:
    pred,scores = label(classes,*c,doc=d['abs'])
    # print pred
    if pred == labels[d['id']]:
      correct += 1
    else:
      errors[labels[d['id']]] +=1
      # print "%s %s" % (d['id'], scores)
  print "Got %s correct of %s" % (correct, test_length)  
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