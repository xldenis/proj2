from math import sqrt
from collections import defaultdict
from feat import *
from multiprocessing import Pool
import random

class KNN:

  def __init__(self, k, x_train, x_label, ):
    self.tfidf = TFIDF(x_train)
    self.k = k
    self.x_train = x_train
    self.x_label = x_label

  def norm(self,doc_idf):
    return sqrt(sum([v*v for v in doc_idf ]))

  def classify(self, knn, labels):
    counts = defaultdict(lambda: 0)
    for k, doc_id in knn:
      counts[labels[doc_id]] += 1
    v = list(counts.values())
    k = list(counts.keys())
    return k[v.index(max(v))]   
     
  def knn(self, query):
    q_idf = self.tfidf.query(query)
    words_q = self.tfidf.strip(query['abs']).split()
    q_norm = self.norm(q_idf.values())

    distances = []
    for doc in self.x_train:
      words_d = self.tfidf.strip(doc['abs']).split()
      common_words = list(set(words_q).intersection(words_d))
      if len(common_words) == 0:
        continue
      d_idf = self.tfidf.tfidf.getrow(doc['id'])
      num = sum([self.tfidf.tfidf[doc['id'],self.tfidf.index(w)] * q_idf[self.tfidf.index(w)] for w in common_words]) 
      distances.append((num / (1+self.norm(d_idf.toarray()[0]) * q_norm), doc['id']))

    return self.classify(sorted(distances)[:self.k],self.x_label)

def main():
  print "kNN test"
  print "Loading Training"
  train2 = load_training()
  print "Loading Labels"
  label = load_label()
  print "Loading Test Data"
  # test  = load_test()
  train, label, true_ids = load_random_subset(1000)
  print "Building KNN"
  knn = KNN(5, train, label)
  for i in random.sample(range(len(train2)), 100):
    randDoc = random.choice(train2)
    print  str(randDoc['id'])+", " + knn.knn(randDoc)
if  __name__ =='__main__':main()