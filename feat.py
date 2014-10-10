import re
from collections import defaultdict
from math import log
import csv
from scipy.sparse import linalg
from scipy.sparse import csr_matrix, lil_matrix, diags
from nltk.stem import PorterStemmer

class TFIDF:
  STOP_WORDS = ["a", "also", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "in", "is", "it", "of", "on", "our", "show", "such", "that", "the", "these", "this", "to", "using", "we", "which", "with"]

  def __init__(self,corpus):
    self.counts = self.count_words(corpus)
    self.tfidf  = self.lsi(self.calc_tfidf(corpus))

  def strip(self, abs):
    ps = PorterStemmer()
    return ' '.join([ ps.stem(w) for w in re.findall('\w+', abs.lower()) if w not in self.STOP_WORDS])

  def count_words(self, corpus):
    print "COUNTING WORDS"
    counts = defaultdict(lambda: 0)
    for doc in corpus:
      for word in self.strip(doc['abs']).split():
        counts[word] += 1
    return counts

  def calc_tfidf(self, corpus):
    print "COUNTING TF"
    tf = {}
    for doc in corpus:
      tf[doc['id']] = defaultdict(lambda: 0)
      for word in self.strip(doc['abs']).split():
        tf[doc['id']][word] += 1
    print "BUILDING TFIDF"
    wordList = sorted(self.counts)
    indices = dict(zip(wordList, range(0, len(wordList))))
    print "USING " + str(len(wordList)) + " WORDS AFTER STEMMING"
    tfidf = lil_matrix((len(corpus), len(wordList)))
    for doc in corpus:
      print doc['id']
      for word in tf[doc['id']]:
        if self.counts[word] >= 3:
          tfidf[int(doc['id']),indices[word]] = tf[doc['id']][word] * log(len(corpus) / float(1 + self.counts[word]))

    return tfidf.tocsr()

  def lsi(self, tfidf):
    print "SELECTING FEATURES"
    u,sigma,vt = linalg.svds(tfidf,k=1000)
    print "SVD Done"
    rows, cols = tfidf.shape
    for index in xrange(rows - 1000, rows):
        sigma[index] = 0
    final = u.dot(diags(sigma,[0])).dot(vt)
    return final

def load_training():
  file = open('train_input.csv','r')
  reader = csv.reader(file, delimiter=',', quotechar='"')
  train = []
  next(reader, None) 
  for row in reader:
    train.append({'id': row[0], 'abs': row[1]})
  return train

def load_label():
  file = open('train_output.csv','r')
  reader = csv.reader(file, delimiter=',', quotechar='"')
  train = {}
  next(reader, None) 
  for row in reader:
    train[row[0]] = row[1]
  return train

def load_test():
  file = open('test_input.csv','r')
  reader = csv.reader(file, delimiter=',', quotechar='"')
  train = []
  next(reader, None) 
  for row in reader:
    train.append({'id': row[0], 'abs': row[1]})
  return train

STOP_WORDS = ["a", "also", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "in", "is", "it", "of", "on", "our", "show", "such", "that", "the", "these", "this", "to", "using", "we", "which", "with"]

def strip(abs):
    ps = PorterStemmer()
    return ' '.join([ ps.stem(w) for w in re.findall('\w+', abs.lower()) if w not in STOP_WORDS])

def main():
  train = load_training()
  print "TF-IDF test"
  # print TFIDF(train)
  counts = defaultdict(lambda: 0)
  for doc in train:
    for word in strip(doc['abs']).split():
      counts[word] += 1
  for word in counts.keys():
    print word
if  __name__ =='__main__':main()