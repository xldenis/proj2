import re
from collections import defaultdict
from math import log
from nltk.stem import PorterStemmer
from util import *

class TFIDF:
  STOP_WORDS = ["a", "also", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "in", "is", "it", "of", "on", "our", "show", "such", "that", "the", "these", "this", "to", "using", "we", "which", "with"]

  def __init__(self,corpus):
    self.counts = self.count_words(corpus)
    self.tfidf  = self.calc_tfidf(corpus)

  def strip(self, abs):
    ps = PorterStemmer()
    return ' '.join([ ps.stem(w) for w in re.findall('\w+', abs.lower()) if w not in self.STOP_WORDS])

  def tf(self, doc):
    freq = defaultdict(lambda: 0)
    for word in self.strip(doc['abs']).split():
      freq[word] += 1
    return freq

  def count_words(self, corpus):
    print "Counting words"
    counts = defaultdict(lambda: 0)
    for doc in corpus:
      for word in self.strip(doc['abs']).split():
        counts[word] += 1
    return counts

  def index(self, word):
    return self.indices[word]

  def query(self, doc):
    tf = self.tf(doc)
    tfidf = {}
    for w in self.strip(doc['abs']).split():
      tfidf[self.index(w)] = tf[w] * log(self.tfidf.shape[0]/ float(1 + self.counts[w]))
    return tfidf

  def calc_tfidf(self, corpus):
    print "Counting TF"
    tf = {}
    for doc in corpus:
      tf[doc['id']] = self.tf(doc)
    print "Building TFIDF"
    wordList = sorted(self.counts)
    self.indices = dict(zip(wordList, range(0, len(wordList))))
    print "Using " + str(len(wordList)) + " words after stemming"
    tfidf = lil_matrix((len(corpus), len(wordList)))
    for doc in corpus:
      # print doc['id']
      for word in tf[doc['id']]:
        if self.counts[word] >= 3:
          tfidf[int(doc['id']), self.indices[word]] = tf[doc['id']][word] * log(len(corpus) / float(1 + self.counts[word]))
    return tfidf.tocsr()

def main():
  train, labels, true_ids = load_random_subset(1000)
  print "TF-IDF test"
  tf = TFIDF(train)
  print TFIDF(train)

if  __name__ =='__main__':main()