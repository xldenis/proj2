from scipy.sparse import linalg
from sparsesvd import sparsesvd
from scipy.sparse import csr_matrix, lil_matrix, diags, hstack
import random
import csv
from collections import Counter

def lsa(tfidf, dim=1500):
  u,sigma,vt = sparsesvd(tfidf.tocsc(),dim)
  rows, cols = tfidf.shape
  sig = diags([sigma],[0],shape=(u.shape[1],vt.shape[0]))
  return csr_matrix(u.dot(sig.dot(vt)))

def simple_select(tfidf, dim=1500):
  mat = tfidf.tfidf.tocsc()
  blocks = []
  trans = {}
  indices = {}
  i = 0
  for w, c in Counter(tfidf.counts).most_common(dim):
    trans[w] = c
    indices[w] = i
    i += 1
    blocks.append(mat.getcol(tfidf.index(w)))
  tfidf.tfidf = hstack(blocks)
  tfidf.indices = indices
  tfidf.counts = trans

def load_training():
  file = open('train_input.csv','r')
  reader = csv.reader(file, delimiter=',', quotechar='"')
  train = []
  next(reader, None) 
  for row in reader:
    train.append({'id': row[0], 'abs': row[1]})
  return train

def load_random_subset(k):
  train = load_training()
  labels = load_label()
  indices = random.sample(range(len(train)), k)
  sample = [train[i] for i in sorted(indices)]
  return ([{'id': i, 'abs': sample[i]['abs']} for i in range(0, len(sample))], [labels[x['id']] for x in sample], [i['id'] for i in sample])

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
