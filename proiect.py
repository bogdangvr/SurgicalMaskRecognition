#Importarea modulelor de care am nevoie

import numpy as np
from scipy.io import wavfile
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from numpy.fft import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import svm

#Citirea datelor train si aplicarea fft pe reverse
train = []
for i in range(1,8001):
    idx = 100000+i
    rate, trainFile = wavfile.read('D:/Workspace/Anul 2/Sem2/ML/train/' + str(idx) + '.wav')
    train.append(np.abs(fft(trainFile[::-1])))

#Citirea datelor validation si aplicarea fft pe reverse
trainValidated = []
for i in range(1,1001):
    idx = 200000+i
    rate, trainValidatedFile = wavfile.read('D:/Workspace/Anul 2/Sem2/ML/validation/' + str(idx) + '.wav')
    trainValidated.append(np.abs(fft(trainValidatedFile[::-1])))
#Concatenarea datelor
trainConcatenated = train + trainValidated

#Citirea labelurilor pentru train si validation
trainLabels = []
trainLabelFile = open('D:/Workspace/Anul 2/Sem2/ML/train.txt', 'r')
trainLabel = trainLabelFile.readlines()
trainLabel.sort()

for lineTrainLabel in trainLabel:
    trainLabelFileName, trainLabelValidation = lineTrainLabel.split(',')
    trainLabels.append(int(trainLabelValidation[0]))
    
    
trainValidatedLabels = []
trainValidatedLabelFile = open('D:/Workspace/Anul 2/Sem2/ML/validation.txt', 'r')
trainValidatedLabel = trainValidatedLabelFile.readlines()
trainValidatedLabel.sort()

for lineTrainValidatedLabel in trainValidatedLabel:
    trainValidatedLabelFileName, trainValidatedLabelValidation = lineTrainValidatedLabel.split(',')
    trainValidatedLabels.append(int(trainValidatedLabelValidation[0]))
#Concatenarea datelor
trainLabelsConcatenated =  trainLabels + trainValidatedLabels


vectorizer = StandardScaler()
X = vectorizer.fit_transform(trainConcatenated)


#Toate modelele testate si scorul obtinut de fiecare:
#clf = svm.SVC(kernel = 'linear', verbose = True) scor 0.638
#clf = MLPClassifier(solver='adam', random_state=1, verbose = True) scor 0.729
#clf = SGDClassifier() scor 0.669
#clf = svm.SVC(kernel = 'poly', verbose = True) scor (si reversed) 0.589 scor reverse 0.678
#clf = KNeighborsClassifier() scor 0.604
#clf = MLPClassifier(solver='lbfgs', random_state=1, verbose = True) scor 0.719
#clf = SGDClassifier(loss = 'perceptron', verbose = True) scor 0.669
#clf = RandomForestClassifier(max_depth = 7, verbose = True) scor 0.668
#clf = RandomForestClassifier(max_depth = 3, verbose = True) scor 0.64
#clf = RandomForestClassifier(max_depth = 10, verbose = True) scor 0.675
#clf = RandomForestClassifier(max_depth = 15, verbose = True) scor 0.672
#clf = MLPClassifier(solver='lbfgs', random_state=1, verbose = True) scor reversed: 0.722
#clf = MLPClassifier(solver='adam', random_state=1, verbose = True) scor reversed: 0.716
#clf = MLPClassifier(solver='adam', random_state=1, verbose = True, learning_rate_init = 0.01, learning_rate = 'adaptive') scor reversed 0.717
#clf = KNeighborsClassifier() scor reverse 0.604
#clf = KNeighborsClassifier(n_neighbors = 10, weights = 'distance') scor reverse 0.617
#clf = KNeighborsClassifier(n_neighbors = 3, weights = 'distance') scor reverse 0.618
#clf = KNeighborsClassifier(weights = 'distance') scor reverse 0.604
#clf = RandomForestClassifier(max_depth = 7, verbose = True) scor reverse 0.654
#clf = GaussianNB() scor reverse 0.592
#clf = DecisionTreeClassifier(random_state=0) scor reverse 0.604
#clf = svm.SVC(C = 10, kernel = 'poly', verbose = True) scor reverse 0.643
#clf = svm.SVC(C = 1000, kernel = 'poly', verbose = True) #scor reverse 0.678
#clf = MLPClassifier(solver='sgd', learning_rate = 'adaptive', random_state=1, verbose = True) scor reverse 0.716
#clf = MLPClassifier(solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.0001, random_state=1, verbose = True) scor reverse 0.725
#clf = MLPClassifier(solver='sgd', learning_rate = 'adaptive', learning_rate_init = 0.00001, random_state=1, verbose = True) scor reverse 0.655
#clf = MLPClassifier(solver='adam', learning_rate = 'adaptive', learning_rate_init = 0.0001, random_state=1, verbose = True) scor reverse 0.727
#clf = MLPClassifier(solver='adam', learning_rate = 'adaptive', learning_rate_init = 0.001, random_state=1, verbose = True) scor reverse 0.716

clf = MLPClassifier(solver='adam', random_state=1, verbose = True)
clf.fit(X, trainLabelsConcatenated)

Y = vectorizer.transform(trainValidated)
predict = clf.predict(Y)
print(accuracy_score(predict, trainValidatedLabels))


validation = []
for i in range(1,1001):
    idx = 200000+i
    fs, validationFile = wavfile.read('D:/Workspace/Anul 2/Sem2/ML/validation/' + str(idx) + '.wav')
    validation.append(np.abs(fft(validationFile[::-1])))

validationLabels = []
validationLabelFile = open('D:/Workspace/Anul 2/Sem2/ML/validation.txt', 'r')
validationLabel = validationLabelFile.readlines()
validationLabel.sort()

for lineValidationLabel in validationLabel:
    validationLabelFileName, validationLabelValue = lineValidationLabel.split(',')
    validationLabels.append(int(validationLabelValue[0]))


X2 = vectorizer.transform(validation)


prediction = clf.predict(X2)

#afisarea scorului
f1_score(validationLabels, prediction, average='macro')

#afisarea confusion matrix
print('Confusion Matrix :')
results = confusion_matrix(validationLabels,prediction)
print(results)
print('Accuracy Score :',accuracy_score(validationLabels,prediction))
print ('Report : ')
print(classification_report(validationLabels,prediction))

#citirea si aplicarea modelului pe teste
test = []
name = []
for i in range(1,3001):
    idx = 300000+i
    fs, testFile = wavfile.read('D:/Workspace/Anul 2/Sem2/ML/test/' + str(idx) + '.wav')
    name.append(str(idx) + '.wav')
    testFileString = testFile
    test.append(np.abs(fft(testFileString[::-1])))

submitFeatures = vectorizer.transform(test)
submitPrediction = clf.predict(submitFeatures)

#scrierea in csv a rezultatelor
with open ('Bogdan_ml.csv', 'w', newline = '') as f:
    filewriter=csv.writer(f)
    filewriter.writerow(['name','label'])
    for i in range(len(submitPrediction)):
        filewriter.writerow([name[i],submitPrediction[i]])