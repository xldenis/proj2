import csv
import numpy as np
from nltk.probability import FreqDist

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm

categories = ['math', 'cs', 'stat', 'physics']
data_location = '/usr/local/data/adoyle/'

n_features = 2500
n_samples = 50000

training_set = []
allwords = []
normalized_words = []
text = ''

#nothing = {}
#bags = [nothing, nothing, nothing, nothing]

#allwords = nothing

train_file = open(data_location + 'train_input.csv', 'rb')
labels_file = open(data_location + 'train_output.csv', 'rb')

training_reader = csv.reader(train_file, delimiter=',', quotechar='"')
label_reader = csv.reader(labels_file, delimiter=',', quotechar='"')


#zip(training_reader, label_reader)
sample = next(training_reader, None)
label = next(label_reader, None)

for i in range(20000):
    sample = next(training_reader, None)
    label = next(label_reader, None)

    text = text + " " + sample[1]
    
train_file.close()
labels_file.close()
    
print 'combined all texts...'
allwords = text.split(" ")
print 'split em up'

for w in allwords:
    if w.isalpha() and len(w) > 3:
        normalized_words.append(w.lower())
    
print 'normalized all words'
fd = FreqDist(normalized_words)
#most_common = fd.items()[1:1000]

most_common = fd.keys()[150:n_features]

prior_keywords = []
for p in most_common:
    prior_keywords.append(p)


print 'finished bagging all the words'

train_file = open(data_location + 'train_input.csv', 'rb')
labels_file = open(data_location + 'train_output.csv', 'rb')

training_reader = csv.reader(train_file, delimiter=',', quotechar='"')
label_reader = csv.reader(labels_file, delimiter=',', quotechar='"')

next(training_reader, None)
next(label_reader, None)

features = np.zeros((n_samples, n_features))
labels = np.zeros(n_samples)

dict_features = []

for i in range(n_samples):
    sample = next(training_reader, None)
    label = next(label_reader, None)
    
    words = sample[1].split(' ')

    keywords = []
    for w in words:
        if w.lower() in prior_keywords:
            keywords.append(w.lower())

    dict_words = dict.fromkeys(most_common, 0)
    
    for w in keywords:
        dict_words[w] = dict_words[w] + 1

    dict_features.append(dict_words)

    j=0
    for key in dict_words.iteritems():
        features[i][j] = key[1]
        j+=1
    
    if label[1] == 'cs':
        labels[i] = 0
    elif label[1] == 'physics':
        labels[i] = 1
    elif label[1] == 'stat':
        labels[i] = 2
    elif label[1] == 'math':
        labels[i] = 3

    
print 'finished counting all the words'


print 'training random forest...'
forest = RandomForestClassifier(n_estimators = 200)
forest.fit(features[0:n_samples/2,:], labels[0:n_samples/2])

print 'training gnb'
bayes = GaussianNB()
bayes.fit(features[0:n_samples/2,:], labels[0:n_samples/2])

print 'training SGD'
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(features[0:n_samples/2, :], labels[0:n_samples/2])

print 'training svm'
svm = svm.SVC()
svm.fit(features[0:n_samples/2, :], labels[0:n_samples/2])

output_forest = forest.predict(features[n_samples/2+1:n_samples])
output_bayes = bayes.predict(features[n_samples/2+1:n_samples])
output_sgd = clf.predict(features[n_samples/2+1:n_samples])
output_svm = svm.predict(features[n_samples/2+1:n_samples])


test = labels[n_samples/2+1:n_samples]
 
right_forest = 0
right_bayes = 0 
right_sgd = 0
right_svm = 0

for i in range(len(output_forest)):
    if output_forest[i] == test[i]:
        right_forest = right_forest + 1
    if output_bayes[i] == test[i]:
	right_bayes = right_bayes + 1
    if output_sgd[i] == test[i]:
	right_sgd = right_sgd + 1
    if output_svm[i] == test[i]:
	right_svm = right_svm + 1


print right_forest
print right_bayes
print right_sgd
print right_svm

# Load real test_set
test_set = []
with open(data_location + 'test_input.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader, None)  # skip the header
    for sample in reader:        
        

        #extract features
        words = sample[1].split(' ')

        keywords = []
        for w in words:
            if w.lower() in prior_keywords:
                keywords.append(w.lower())

        dict_words = dict.fromkeys(most_common, 0)
    
        for w in keywords:
            dict_words[w] = dict_words[w] + 1

        dict_features.append(dict_words)

        j=0
        test_features = np.zeros(n_features)
        for key in dict_words.iteritems():
            test_features[j] = key[1]
            j+=1
            
        test_set.append([sample, test_features])
        

# Write a random category to the csv file for each example in test_set
output_file = open(data_location + 'test_output.csv', "wb")
writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL) 

writer.writerow(['id', 'category']) # write header
for sample in test_set:
    prediction = forest.predict(sample[1])
    
    if prediction == 0:
        category = 'cs'
    elif prediction == 1:
        category = 'physics'
    elif prediction == 2:
        category = 'stat'
    elif prediction == 3:
        category = 'math'

    row = [sample[0][0], category]
    writer.writerow(row)

output_file.close()
