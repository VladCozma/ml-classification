# libraries import        
import pandas as pd
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

# load the trainset
trainset = pd.read_excel("training-dataset.xlsx")
#print('Training dataset loaded.\n')
#print('Training dataset description:\n')
#trainset.info()
#print('\n')

# Load the validation set
validationset = pd.read_csv("validation-dataset.csv",sep=";")
#print('Validation dataset loaded.\n')
#print('Validation dataset description:\n')
#validationset.info()
#print('\n')

#from sklearn.model_selection  import train_test_split
# this method split one dataset into train set and validation set
# I don't need this, since I have train set and validation set
#    X_train, X_test, y_train, y_test = train_test_split(trainset.Message, trainset.Label, test_size=0.25, random_state=33)
 
# a pipeline build

# Create and tweak vectorizer
# vectorize only on English words
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
# filter more important words
min_occ=2
#
#import string
#from nltk.stem import PorterStemmer
#from nltk import word_tokenize
# 
#def stemming_tokenizer(text):
#    stemmer = PorterStemmer()
#    return [stemmer.stem(w) for w in word_tokenize(text)]

# final vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), min_df=min_occ)

# Create and tweak classifier
# various classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
# First classifier to be used in Multinomial Bayes
#set alpha value for the Bayes classifier
alpha_value=0.02

classifier_model = MultinomialNB(alpha=alpha_value, fit_prior=False)

# training pipeline
from sklearn.pipeline import Pipeline 
training = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier_model),
])
    
# training the model
# = train(training, trainset.Message, trainset.Label)
classifier = training.fit(trainset.Message, trainset.Label)

# writing predictions
print("Accuracy: %s" % classifier.score(validationset.Message, validationset.Label))
output = pd.DataFrame(classifier.predict(validationset.Message))
output = pd.concat([validationset.Message, output],axis=1)
with open('validation-results.csv', 'w+') as f:        
    output.to_csv(f, header=True, sep=';')    

# saving the model
import pickle
s = pickle.dumps(classifier)
    
# mode to come on persistence    
#nltk.download(joblib)
#from joblib import dump, load
#dump(classifier, 'classifier_trained.joblib')
# load('classifier_trained.joblib.joblib')
#print(output)





# The above configuration gives an accuracy of about 0.71. I will try to iterate over various classifier to see if I can get it higher
##
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=0.1),
    AdaBoostClassifier(),
    MultinomialNB(alpha=alpha_value, fit_prior=False)]
import array as arr
accuracy = arr.array('d',[])
i = -1
for name, clf in zip(names, classifiers):
    print("Training for %s." % name)
    training = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', clf),
        ])
    classifier = training.fit(trainset.Message, trainset.Label)     
    accuracy.append(classifier.score(validationset.Message, validationset.Label))
    print("Accuracy for %s is %s" % (name, accuracy[++i]))

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
y_pos = np.arange(len(names))
 
plt.bar(y_pos, accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, names, rotation='vertical', verticalalignment='bottom')
plt.ylabel('Accuracy')
plt.title('Classifier')
plt.figure(figsize=(20000,20000)) 
plt.show() 
    
#    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#    clf.fit(X_train, y_train)
#    score = clf.score(X_test, y_test)
#
#    # Plot the decision boundary. For that, we will assign a color to each
#    # point in the mesh [x_min, x_max]x[y_min, y_max].
#    if hasattr(clf, "decision_function"):
#        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#    else:
#        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
#    # Put the result into a color plot
#    Z = Z.reshape(xx.shape)
#    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
#
#    # Plot the training points
#    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
#               edgecolors='k')
#    # Plot the testing points
#    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
#               edgecolors='k', alpha=0.6)
#
#    ax.set_xlim(xx.min(), xx.max())
#    ax.set_ylim(yy.min(), yy.max())
#    ax.set_xticks(())
#    ax.set_yticks(())
#    if ds_cnt == 0:
#        ax.set_title(name)
#    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#            size=15, horizontalalignment='right')
#    i += 1
