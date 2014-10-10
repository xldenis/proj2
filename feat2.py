import re
from collections import defaultdict
from math import log
import csv, string
import operator
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm


allwords = defaultdict(lambda: 0)
n_features = 1000

counts = defaultdict(lambda: 0)
training_samples = 0

feature_list = []

def strip(abs):
  return ' '.join(re.split('\W+', abs))

def tfidf(corpus):
  tf = {}
  training_samples = len(corpus)

  for doc in corpus:
    tf[doc['id']] = defaultdict(lambda: 0)
    for word in strip(doc['abs']).split():
      w = word.lower()
      allwords[w] += 1
      tf[doc['id']][w] += 1
      counts[w] += 1
  tfidf = {}

  for doc in corpus:
    tfidf[doc['id']] = defaultdict(lambda: 0)
    for word in tf[doc['id']]:
      tfidf[doc['id']][word] = tf[doc['id']][word] * log(len(corpus) / float(1 + counts[word]))
  return tfidf

def test_samples(corpus):
  i=0
  for doc in corpus:
    word_count = np.zeros(n_features)
    feature_vector = np.zeros((len(corpus), n_features))
    for word in strip(doc['abs'].split()):
      w = word.lower()
      if w in feature_list:
        index = feature_list.index(w)
        word_count[index]+=1
    
    for j in range(n_features-1):
      feature_vector[i][j] = word_count[j] * log(training_samples / float(1 + word_count[j]))
    i+=1

  return feature_vector


def load_training():
  
  data_file = open('train_input.csv','r')
  label_file = open('train_output.csv','r')

  data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
  label_reader = csv.reader(label_file, delimiter=',', quotechar='"')
  train = []
  train_label = []

  for row, label in zip(data_reader, label_reader):
    if label[1] in ['math', 'cs', 'stat', 'physics']:
      train.append({'id': row[0], 'abs': row[1]})
      train_label.append(label[1])

  return train, train_label

def load_test():
  file = open('test_input.csv', 'r')
  reader = csv.reader(file, delimiter=',', quotechar='"')

  test = []
  for row in reader:
    test.append({'id': row[0], 'abs': row[1]})
  
  return test



def main():
  train, train_label = load_training()
  
  print "TF-IDF test"
  features = tfidf(train)

  for f in features:
    for w in features[f].items():
      if w[1] > allwords[w[0]]:
        allwords[w[0]] = w[1]

  for n in range(n_features):
    feature_word = max(allwords, key=allwords.get)
    feature_list.append(feature_word)
    allwords[feature_word] = -10


  feats = np.zeros((len(features), n_features))

  i = 0
  for f in features:
    for w in features[f].items():
      if w[0] in feature_list:
	index = feature_list.index(w[0])
        feats[i,index] = w[1]
    i+=1


  print 'training forest...'
  forest = RandomForestClassifier(n_estimators=100)
  forest.fit(feats[0:10, :], train_label[0:10])  
  print 'done. testing...'

  test = load_test()
  print test
  test_features = test_samples(test)

  output = forest.predict(test_features)

  output_file = open('test_output.csv', "wb")
  writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

  writer.writerow(['id', 'category'])
  i=0
  for t in test:
    row = [t['id'], output[i]]
    writer.writerow(row)
    i+=1
  output_file.close()

  return 0

if  __name__ =='__main__':main()
