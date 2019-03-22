# libraries import        
import pandas as pd

# for text vectorization
import nltk
nltk.download('stopwords')

# load the trainset
trainset = pd.read_csv("training-dataset.csv",sep=";")
print('Training dataset loaded. Description below:\n')
#trainset.info()
#print('\n')

# Load the validation set
validationset = pd.read_csv("validation-dataset.csv",sep=";")
print('Validation dataset loaded. Description below:\n')
#validationset.info()
#print('\n')

## to split one dataset in training set and validation set, train_test_split can be used 
# from sklearn.model_selection  import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(trainset.Message, trainset.Label, test_size=0.25, random_state=33)
 

# train method
def train(classifier, message, label):
    classifier.fit(trainset.Message, trainset.Label)
    print("Accuracy: %s" % classifier.score(validationset.Message, validationset.Label))
    return classifier	

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline 

# a pipeline build

# Tweak vectorizer
# vectorize only on English words
from nltk.corpus import stopwords
# filter more important words
min_occ=2

# Tweak classifier
#set alpha value for the Bayes classifier
alpha_value=0.02

trial = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'), min_df=min_occ)),
    ('classifier', MultinomialNB(alpha=alpha_value)),
])
    
# training the model
classifier = train(trial, trainset.Message, trainset.Label)

# writing predictions
output = pd.DataFrame(classifier.predict(validationset.Message))
output = pd.concat([validationset.Message, output],axis=1)

with open('validation-results.csv', 'w+') as f:        
    output.to_csv(f, header=True, sep=';')    

#print(output)

