import re
from collections import defaultdict
from math import log
import csv

def strip(abs):
  return ' '.join(re.split('\W+', abs))

def tfidf(corpus):
  tf = {}
  counts = defaultdict(lambda: 0)

  for doc in corpus:
    tf[doc['id']] = defaultdict(lambda: 0)
    for word in strip(doc['abs']).split():
      tf[doc['id']][word] += 1
      counts[word] += 1
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
  print tfidf(train)

if  __name__ =='__main__':main()