from math import log
from collections import defaultdict
from util import *

def strip(doc):
  return doc

def vocab(corpus):
  words = set()
  for d in corpus:
    for t in strip(d).split():
      set.add(t)
  return set

def train(classes, corpus): 
  vocab = vocab(corpus)
  prior = {}
  probs = {}
  for c in classes:
    prior[c] = 0
    for doc in c:
      prioc[c] += 1
      for each t in strip(doc).split():
        if t in vocab:
          counts[c][t] += 1
    prior[c] /= len(corpus)
    totalCount = sum([counts[c][t] +1 for t in vocab])
    probs[c] = defaultdict(lambda: 1.0 / len(terms))
    for t in vocab:
      if count[c][t] > 0:
        probs[c][t] = counts[c][t] + 1 / float(totalCount - (counts[c][t] + 1))

  return v, prior, probs

def label(classes, vocab, prior, probs, doc):
  words = [t for t in strip(doc).split() if t in vocab]
  for c in classes:
    score[c] = log(prior[c])
    for t in words:
      score[c] += log(probs[c][t])
  return classes(score.index(max(score)))


def main:
  classes = ['physics', 'cs', 'math', 'stats']

  training = load_training()
  corpus = [d['abs'] for d in training]

  vocab, prior, probs = train(classes, corpus)

if  __name__ =='__main__':main()