from math import sqrt
from collections import defaultdict
from feat import *
import random

class KNN:

  def __init__(self, k, x_train, x_label, ):
    self.tfidf = TFIDF(x_train)
    self.k = k
    self.x_train = x_train
    self.x_label = x_label

  def sim(self,x,d):
    words_x = self.tfidf.strip(x['abs']).split()
    words_d = self.tfidf.strip(x['abs']).split()
    common_words = list(set(words_x).intersection(words_d))
    if len(common_words) == 0:
      return 0.0
    d_idf = self.tfidf.tfidf[d['id']]
    x_idf = self.tfidf.query(x)
    num = sum([d_idf[self.tfidf.index(w)] * x_idf[self.tfidf.index(w)] for w in common_words]) 
    return num / (1+self.norm(d) * self.norm(x))

  def norm(self,doc):
    return sqrt(sum([v*v for _k,v in self.tfidf.query(doc).iteritems() ]))

  def classify(self, knn, labels):
    counts = defaultdict(lambda: 0)
    for k, doc_id in knn:
      counts[labels[doc_id]] += 1
    v = list(counts.values())
    k = list(counts.keys())
    return k[v.index(max(v))]   
     
  def knn(self, query):
    x_idf = self.tfidf.query(query)
    words_q = self.tfidf.strip(query['abs']).split()
    x_norm = self.norm(x)

    distances = []
    for doc in self.x_train:
      words_d = self.tfidf.strip(x['abs']).split()
      common_words = list(set(words_x).intersection(words_d))
      if len(common_words) == 0:
        continue
      d_idf = self.tfidf.tfidf[d['id']]
      num = sum([d_idf[self.tfidf.index(w)] * x_idf[self.tfidf.index(w)] for w in common_words]) 
      distances.append((num, doc['id']))

    distances = [(self.sim(query, x), x['id']) for x in self.x_train]
    return self.classify(sorted(distances)[:self.k],self.x_label)

def main():
  print "kNN test"
  print "Loading Training"
  train = load_training()
  print "Loading Labels"
  label = load_label()
  print "Loading Test Data"
  # test  = load_test()
  train, label, true_ids = load_random_subset(1000)
  print "Building KNN"
  knn = KNN(5, train, label)
  for i in range(0,10):
    randDoc = random.choice(train)
    print  str(true_ids[randDoc['id']])+", " + knn.knn(randDoc)
if  __name__ =='__main__':main()