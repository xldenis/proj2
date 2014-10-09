from math import sqrt
from collections import defaultdict
from feat import *
import random

class KNN:

  def __init__(self, k, x_train, x_label, ):
    self.tfidf = tfidf(x_train)
    self.k = k
    self.x_train = x_train
    self.x_label = x_label

  def sim(self,x,d):
    words_x = strip(x['abs'])
    words_d = strip(x['abs'])
    common_words = list(set(words_x).intersection(words_d))
    num = sum([self.tfidf[d['id']][w] * self.tfidf[x['id']][w] for w in common_words]) 
    return num / (1+self.norm(d) * self.norm(x))


  def norm(self,doc):
    return sqrt(sum([v*v for _k,v in self.tfidf[doc['id']].iteritems() ]))

  def classify(self, knn, labels):
    counts = defaultdict(lambda: 0)
    for k, doc_id in knn:
      counts[labels[doc_id]] += 1
    v = list(counts.values())
    k = list(counts.keys())
    return k[v.index(max(v))]   
     
  def knn(self, query):
    distances = [(self.sim(query, x), x['id']) for x in self.x_train]
    return self.classify(sorted(distances)[:self.k],self.x_label)

def main():
  print "kNN test"
  print "Loading Training"
  train = load_training()
  print "Loading Labels"
  label = load_label()
  print "Loading Test Data"
  test  = load_test()
  print "Building KNN"
  knn = KNN(5, train, label)
  randDoc = random.choice(train)
  print "Chose" + randDoc['id']
  print "OUTPUT" + knn.knn(randDoc)
if  __name__ =='__main__':main()