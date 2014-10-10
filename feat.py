import re
from collections import defaultdict
from math import log
import csv

STOP_WORDS = ["a", "also", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "in", "is", "it", "of", "on", "our", "show", "such", "that", "the", "these", "this", "to", "using", "we", "which", "with"]

def strip(abs):
  return ' '.join([w.lower() for w in re.findall('\w+', abs) if w not in STOP_WORDS])

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
      if counts[word] >= 3:
        tfidf[doc['id']][word] = tf[doc['id']][word] * log(len(corpus) / float(1 + counts[word]))

  return tfidf

def load_training():
  file = open('train_input.csv','r')
  reader = csv.reader(file, delimiter=',', quotechar='"')
  train = []
  for row in reader:
    train.append({'id': row[0], 'abs': row[1]})
  return train

def load_label():
  file = open('train_output.csv','r')
  reader = csv.reader(file, delimiter=',', quotechar='"')
  train = {}
  for row in reader:
    train[row[0]] = row[1]
  return train

def load_test():
  file = open('test_input.csv','r')
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