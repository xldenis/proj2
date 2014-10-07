import csv
import numpy as np
from nltk.probability import FreqDist
from sklearn.ensemble import RandomForestClassifier

categories = ['math', 'cs', 'stat', 'physics']


n_features = 1000
n_samples = 20000

training_set = []
allwords = []
normalized_words = []
text = ''

#nothing = {}
#bags = [nothing, nothing, nothing, nothing]

#allwords = nothing

train_file = open('/data/train_input.csv', 'rb')
labels_file = open('/data/train_output.csv', 'rb')

training_reader = csv.reader(train_file, delimiter=',', quotechar='"')
label_reader = csv.reader(labels_file, delimiter=',', quotechar='"')


#zip(training_reader, label_reader)
sample = next(training_reader, None)
label = next(label_reader, None)

for i in range(15000):
    sample = next(training_reader, None)
    label = next(label_reader, None)

    text = text + " " + sample[1]
    
train_file.close()
labels_file.close()
    
print 'combined all texts...'
allwords = text.split(" ")
print 'split em up'

for w in allwords:
    if w.isalpha():
        normalized_words.append(w.lower())
    
print 'normalized all words'
fd = FreqDist(normalized_words)
#most_common = fd.items()[1:1000]

most_common = fd.keys()[0:n_features]

prior_keywords = []
for p in most_common:
    prior_keywords.append(p)


print 'finished bagging all the words'

train_file = open('/data/train_input.csv', 'rb')
labels_file = open('/data/train_output.csv', 'rb')

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
forest = RandomForestClassifier(n_estimators = 200)

print 'training random forest...'
forest.fit(features[0:n_samples,:], labels[0:n_samples])
# output = forest.predict(features[10001:20000])
# 
# test = labels[10001:20000]
# 
# right = 0
# 
# for i in range(len(output)):
#     if output[i] == labels[10001+i]:
#         right = right + 1
# 
# 
# 
# print right
# print len(output)

# Load test_set
test_set = []
with open('/data/test_input.csv', 'rb') as csvfile:
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
output_file = open('/data/test_output.csv', "wb")
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
