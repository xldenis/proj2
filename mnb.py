from math import log
from collections import defaultdict
from util import *
import random
from nltk.stem import PorterStemmer
import re

def strip(doc):
  ps = PorterStemmer()
  # words = re.findall('[\w-]+', re.sub('\$.*?\$','math', doc.lower()))
  # words = re.findall('(\$.*?\$|[\w-]+)', doc.lower())
  words = re.findall('([\w-]+)', doc.lower())
  return [ps.stem(w) for w in words]

def _vocab(corpus):
  words = set([])
  df = Counter()
  for d in corpus:
    wrds = strip(d)
    for w in set(wrds):
      df[w] += 1
    for t in wrds:
      if len(t) > 4:
        words.add(t)
  return (words, df)

def train(classes, corpus, corpus_labels): 
  print 'Extracting vocab'
  vocab, _df = _vocab(corpus)
  print len(vocab)
  prior, probs, counts  = ({}, {}, {})
  for c in classes: 
    print "Learning %s" % c
    prior[c] = 0
    counts[c] = defaultdict(lambda: 0)
    for doc in zip(corpus, corpus_labels):
      if doc[1] == c:
        prior[c] += 1
        for t in strip(doc[0]):
          if t in vocab:
            counts[c][t] += 1

    prior[c] /= float(len(corpus))
    totalCount = sum([counts[c][t] +1 for t in vocab])
    probs[c] = defaultdict(lambda: 1.0 / len(vocab))

    for t in list(vocab):
      if counts[c][t] > 0:  
        probs[c][t] = counts[c][t] + 1 / float(totalCount - (counts[c][t] + 1))

  return vocab, prior, probs

def tfidf(w, f, df, d):
  return log(f[w]/float(len(f)) + 1.0) * log( float(d) / df[w])

def train_with_tfidf(classes, corpus, corpus_labels):
  print 'Extracting vocab'
  vocab, df = _vocab(corpus)
  d = len(corpus)
  print len(vocab)
  prior, probs, tf  = ({}, {}, {})
  for c in classes: 
    print "Learning %s" % c
    prior[c] = 0
    tf[c] = defaultdict(lambda: 0)
    for doc in zip(corpus, corpus_labels):
      if doc[1] == c:
        prior[c] += 1
        for t in strip(doc[0]):
          if t in vocab:
            tf[c][t] += 1

    prior[c] /= float(len(corpus))

    totalCount = sum([tfidf(t,tf[c],df,d) +1 for t in vocab])
    probs[c] = defaultdict(lambda: 1.0 / len(vocab))

    for t in list(vocab):
      if tf[c][t] > 0:  
        probs[c][t] = tfidf(t,tf[c],df,d) + 1 / float(totalCount - (tfidf(t,tf[c],df,d) + 1))

  return vocab, prior, probs, tf, df, d


def label_with_tfidf(classes, vocab, prior, probs, f, df, d, doc):
  words = [t for t in strip(doc) if t in vocab]
  score = {}
  for c in classes:
    score[c] = log(prior[c])
    for t in words:
      score[c] += log(probs[c][t]) 
  return max(score, key=score.get)

def label(classes, vocab, prior, probs, doc):
  words = [t for t in strip(doc) if t in vocab]
  score = {}
  for c in classes:
    score[c] = log(prior[c])
    for t in words:
      score[c] += log(probs[c][t])
  return max(score, key=score.get)


def main():
  classes = ['physics', 'cs', 'math', 'stat']

  total = load_training()
  random.shuffle(total)
  labels = load_label()
  training = total[:len(total)/2][:5000]
  test = total[len(total)/2:]

  print 'Extracting Corpus'
  corpus = [d['abs'] for d in training]
  corpus_labels = [labels[d['id']] for d in training]
  print 'Training'
  c1 = train(classes, corpus, corpus_labels)
  c2 = train_with_tfidf(classes, corpus, corpus_labels)
  compare(c1, c2, classes, test, labels)

def compare(c1,c2, classes, test, labels):
  print 'Testing'
  correct = [0,0]
  both_wrong = 0
  one_right  = 0
  errors = [defaultdict(lambda:0),defaultdict(lambda:0)]
  test_length = 500
  for d in test[:test_length]:
    pred = label(classes,*c1,doc=d['abs'])
    if pred == labels[d['id']]:
      correct[0] += 1
    else:
      errors[0][labels[d['id']]] +=1
    pred2 = label_with_tfidf(classes, *c2, doc=d['abs'])
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


if  __name__ =='__main__':main()