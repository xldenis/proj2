from collections import defaultdict
from math import log
import util
import feat
import random 

def weights(classes, labels, features, tfidf):
  w = {}
  for c in classes:
    print "DENOMINATOR FOR CLASS %s" % c
    denom = 0.0
    i = 0
    for lab in labels:
      if lab != c:
        for feat in features:
          denom += tfidf.tfidf[i, tfidf.index(feat)]
      i +=1
    print "NUMERATOR FOR CLASS %s" % c
    for feat in features:
      w[c] = {}
      i = 0
      for lab in labels:
        if lab != c:
          w[c][feat] = log((tfidf.tfidf[i,tfidf.index(feat)] + 1)/ denom)
        i +=1
  return w

def labels(classes, weights, tf, doc):
  counts = defaultdict(lambda: 0)
  for w in tf.strip(doc['abs']).split():
    counts[w] += 1

  scores = []
  for c in classes:
    for i in tf.indices.keys():
      scores.append(counts[i]*weights[c].get(i,0))

  return classes[scores.index(min(scores))]

def main():
  data, label, ids = util.load_random_subset(1000)
  train2 = util.load_training()
  classes = ['physics', 'cs', 'math', 'stats']

  tf = feat.TFIDF(data)
  util.simple_select(tf)
  w = weights(classes, label, tf.indices.keys(), tf)

  for i in random.sample(range(len(train2)), 100):
    randDoc = random.choice(train2)
    print  str(randDoc['id'])+", " + labels(classes,w,tf,randDoc)

if  __name__ =='__main__':main()