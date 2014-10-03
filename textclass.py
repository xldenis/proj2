import csv
from sklearn.ensemble import RandomForestClassifier

categories = ['math', 'cs', 'stat', 'physics']

training_set = []
words = []

nothing = {}
#bags = [nothing, nothing, nothing, nothing]

allwords = nothing

train_file = open('/data/train_input.csv', 'rb')
labels_file = open('/data/train_output.csv', 'rb')

training_reader = csv.reader(train_file, delimiter=',', quotechar='"')
label_reader = csv.reader(labels_file, delimiter=',', quotechar='"')


for sample, label in zip(training_reader, label_reader):
    next(training_reader, None)
    next(label_reader, None)
    words = sample[1].split(' ')

    for w in words:
        allwords[w] = allwords.get(w,0)+1

train_file.close()
labels_file.close()

print 'finished bagging all the words'

train_file = open('/data/train_input.csv', 'rb')
labels_file = open('/data/train_output.csv', 'rb')

training_reader = csv.reader(train_file, delimiter=',', quotechar='"')
label_reader = csv.reader(labels_file, delimiter=',', quotechar='"')

# bags[0] = dict.fromkeys(allwords, 0)
# bags[1] = dict.fromkeys(allwords, 0)
# bags[2] = dict.fromkeys(allwords, 0)
# bags[3] = dict.fromkeys(allwords, 0)

features = []
labels = []
i = 0

for sample, label in zip(training_reader, label_reader):
    next(training_reader, None)
    next(label_reader, None)
        
    words = sample[1].split(' ')
    
    features.append(dict.fromkeys(allwords, 0))
    labels.append(label[1])
    
    for w in words:
        features[i][w] = features[0].get(w,0)+1
    
    i = i + 1
#         if label[1] == 'math':
#             bags[0][w] = bags[0].get(w,0)+1
#         elif label[1] == 'cs':
#             bags[1][w] = bags[1].get(w,0)+1
#         elif label[1] == 'stat':
#             bags[2][w] = bags[2].get(w,0)+1
#         elif label[1] == 'physics':
#             bags[3][w] = bags[3].get(w,0)+1

print 'finished counting all the words'
forest = RandomForestClassifier(n_estimators = 100)

print 'training random forest...'
forest.fit(features, labels)

print 'extracting test features'

test_file = open('/data/test_input.csv', 'rb')
test_labels = open('/data/test_output.csv', 'rb')

test_reader = csv.reader(test_file, delimiter=',', quotechar='"')
test_label_reader = csv.reader(test_labels, delimiter=',', quotechar='"')

test_features = []
test_labels = []

for sample, label in zip(test_reader, test_label_reader):
    next(test_reader, None)
    next(test_label_reader, None)
        
    words = sample[1].split(' ')
    
    test_features.append(dict.fromkeys(allwords, 0))
    test_labels.append(label[1])
    
    for w in words:
        test_features[i][w] = features[0].get(w,0)+1
    
    i = i + 1

print 'done extracting test features'

predicted = forest.fit(test_features)

total = 0
correct = 0

for p, s in zip(predicted, test_labels):
    total = total + 1
    if s == p:
        correct = correct + 1
        
print 'accuracy: {0}'.format(correct/total)