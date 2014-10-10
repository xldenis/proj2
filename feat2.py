import re
from collections import defaultdict
from math import log
import csv, string
import operator
import numpy as np

allwords = defaultdict(lambda: 0)
n_features = 5000


def strip(abs):
  return ' '.join(re.split('\W+', abs))

def tfidf(corpus):
  tf = {}
  counts = defaultdict(lambda: 0)

  for doc in corpus:
    tf[doc['id']] = defaultdict(lambda: 0)
    for word in strip(doc['abs']).split():
      w = word.lower()
      allwords[w] += 1
      tf[doc['id']][w] += 1
      counts[w] += 1
  tfidf = {}

  for doc in corpus:
    tfidf[doc['id']] = defaultdict(lambda: 0)
    for word in tf[doc['id']]:
      tfidf[doc['id']][word] = tf[doc['id']][word] * log(len(corpus) / float(1 + counts[word]))
  return tfidf

def load_training():
  file = open('train_input.csv','r')
  reader = csv.reader(file, delimiter=',', quotechar='"')
  train = []
  for row in reader:
    train.append({'id': row[0], 'abs': row[1]})
  return train

def main():
  train = load_training()
  print "TF-IDF test"
  features = tfidf(train)

  for f in features:
    for w in features[f].items():
      if w[1] > allwords[w[0]]:
        allwords[w[0]] = w[1]
  
  
  
  feature_list = []

  for n in range(n_features):
    feature_word = max(allwords, key=allwords.get)
    feature_list.append(feature_word)
    allwords[feature_word] = -10


  feats = np.zeros((len(features), n_features))

  i = 0
  for f in features:
    for w in features[f].items():
      if w[0] in feature_list:
	index = feature_list.index(w[0])
        feats[i,index] = w[1]
    i+=1


  return feats

if  __name__ =='__main__':main()
