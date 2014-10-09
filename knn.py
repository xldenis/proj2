from math import math
from collections import defaultdict

def sim(x,d):
  common_words = tfidf[x['id']].keys().intersection(d['id'].keys())
  num = sum([tfidf[x['id']][w] * tfidf[x['id']][w] for x in common_words]) 
  return num / (norm(d) * norm(x))


def norm(doc):
  return sqrt(sum([x**2 for x in tfidf[doc['id']] ]))

def classify(knn,labels):
  counts = defaultdict(lambda: 0)
  for k in knn
    counts[labels[k['id']]] += 1
  return max(counts, key=lambda k: counts[k])
  
def knn(k, x_train, x_label, query):
   distances = [sim(query, x) for x in dataset]
  return classify(sorted(distances)[:k],x_label)