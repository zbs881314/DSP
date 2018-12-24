# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Input data into the computer
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
full = train.append( test , ignore_index = True )
combine = [train, test]
whole = full[ :891 ]

# Description

# Preview the data
print(train.head().to_string())
print(train.tail().to_string())

train.info()
print('_'*40)

print(train.describe(include=['O']).to_string())

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , whole.shape)

# Easy diagram(chart): the code that is embeded in the computer and easy to call
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False).to_string())

print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False).to_string())

train_organized=train.groupby('Survived').agg({'Pclass': {'1': lambda x: (x == 1).sum(),'2': lambda x: (x == 2).sum(),'3': lambda x: (x == 3).sum(), },
                                               'Sex': {'Male': lambda x: (x == 'male').sum(),'Female': lambda x: (x == 'female').sum(), },
                                               'Embarked': {'S': lambda x: (x == 'S').sum(),'C': lambda x: (x == 'C').sum(),'Q': lambda x: (x == 'Q').sum()},
                                               'Age': {'Median': 'median', 'Mean': 'mean', },
                                               'Fare':{'Median': 'median','Mean': 'mean', },
                                               'SibSp': {'Mean': 'mean', },
                                               'Parch': {'Mean': 'mean', },
                                              }
                                             )
print(train_organized.to_string())

# Easy diagram(diagram)

plt.figure()
a = sns.FacetGrid(train, col='Survived')
a.map(plt.hist, 'Age', bins=20)
plt.savefig("Age related Survived")

plt.figure()
b = sns.FacetGrid(train, col='Survived')
b.map(plt.hist, 'Pclass', bins=20)
plt.savefig("Pclass related Survived")

# Red is survived and green is dead
plt.figure()
plot_fare = plt.hist([train[train['Survived']==1]['Fare'].fillna(-10),
          train[train['Survived']==0]['Fare'].fillna(-10)],
         stacked=True, color = ['r','g'],
         bins = 30,label = ['Survived','Dead'])
plt.savefig("Survival related Fare")

plt.figure()
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.4, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.savefig("Pclass&Age related Survived")

plt.figure()
grid = sns.FacetGrid(train, row='Pclass', size=2.4, aspect=1.5)
grid.map(sns.pointplot, 'Embarked', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.savefig("Pclass&Embarked related Survive")

plt.figure()
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.4, aspect=1.5)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.savefig("Embarked&Sex related Survived")

#Hard Diagram: We need to give defination to the function
def plot_correlation_map( df ):
    corr = whole.corr()
    _ , ax = plt.subplots( figsize =( 10 , 12 ) )
    cmap = sns.diverging_palette( 200 , 800 , as_cmap = True )
    _ = sns.heatmap(
        corr,
        cmap = cmap,
        square=True,
        cbar_kws={ 'shrink' : .9 },
        ax=ax,
        annot = True,
        annot_kws = { 'fontsize' : 12 }
    )
plt.figure()
plot_correlation_map( combine )
plt.savefig("Test Correlation")


def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=5 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
plt.figure()
plot_distribution(whole,var = 'Age',target = 'Sex', row = 'Survived')
plt.savefig("Sex&Survived Related Age")

# Forecast Procedure
# reorganized the data
sex = pd.Series( np.where( full.Sex == 'male' , 0 , 1 ) , name = 'Sex' )

embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )

pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

fill_value = pd.DataFrame()
fill_value[ 'Fare' ] = full.Age.fillna( full.Age.mean() )
fill_value[ 'Age' ] = full.Fare.fillna( full.Fare.mean() )
print(fill_value.head().to_string())

cab = pd.DataFrame()
cab[ 'Cabin' ] = full.Cabin.fillna( 'Im' )
cab[ 'Cabin' ] = cab[ 'Cabin' ].map( lambda x : x[0] )
cabin = pd.get_dummies( cab['Cabin'] , prefix = 'Cabin' )
print(cabin.head().to_string())


done = pd.concat( [ fill_value , pclass, embarked, cabin , sex ] , axis=1 )
print(done.head().to_string())

#Organize the data
train_X = done [ 0:891 ]
train_y = whole.Survived
X_test = done[ 891: ]
Xtrain , Xvalid , Ytrain , Yvalid = train_test_split( train_X , train_y , train_size = .7 )

# Find which method should we use for forecasting.
models = []
models.append(( "LR" , LogisticRegression()))
models.append(( "SVM" , SVC()))
models.append(( "NB" , GaussianNB()))
models.append(( "KNN" , KNeighborsClassifier()))
models.append(( "CART" , DecisionTreeClassifier()))



num_folds = 10
seed = 7
scoring = "accuracy"

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, Xtrain, Ytrain, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print (msg)

#Also can be done by the below code
#print(model.score(Xtrain,Ytrain), model.score(Xvalid,Yvalid))

# Using logisticRegression() to show the result
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
Y_pred = model.predict(X_test)
print(model.fit(Xtrain, Ytrain))
print(model.score(Xtrain,Ytrain), model.score(Xvalid,Yvalid))

# Save the prediction result into the excel
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': Y_pred } )
print(test.shape)
print(test.head(10))
test.to_csv( 'titanic_prediction.csv' , index = False )