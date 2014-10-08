import re
from collections import defaultdict
from math import log

def strip(abs):
  stripped = ' '.join([re.match(r'(\w+)' for x in abs.split()])

def calc_tf(corpus):
  dict = {}
  for doc in corpus:
    dict[doc.id] = defaultdict(lambda: 0)
    for word in strip(doc).split():
      dict[doc.id][word] += 1
  dict

def tfidf(corpus):
  tf = {}
  counts = defaultdict(lambda: 0)
  for doc in corpus:
    tf[doc.id] = defaultdict(lambda: 0)
    for word in strip(doc).split():
      tf[doc.id][word] += 1
      count[word] += 1
  tfidf = {}
  for doc in corpus:
    tfidf[doc.id] = defaultdict(lambda: 0)
    for word in tf[doc.id]:
      tfidf[doc.id][word] = tf[doc.id][word] * log(len(corpus) / counts[word])
  tfidf

