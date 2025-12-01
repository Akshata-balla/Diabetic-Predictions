# Python script converted from Classification Project.ipynb

# ==================================
# 1. SETUP AND LOAD LIBRARIES
# ==================================

# Import all necessary libraries
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from numpy import set_printoptions
from numpy import array
import numpy
from matplotlib import colors
from matplotlib import cm as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from pickle import dump
from pickle import load

# ==================================
# 2. LOAD THE DATA
# ==================================

# Load CSV using Pandas
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print(data.shape)

# Load the data again (as done in the original notebook for descriptive stats)
filename = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# ==================================
# 3. DESCRIPTIVE STATS
# ==================================

# Data Types for Each Attribute
types = data.dtypes
print(types)

# Statistical Summary
data.describe()
# set_option('precision', 3)

# description = data.describe()
# print(description)

# Pairwise Pearson correlations
correlations = data.corr(method='pearson')
print(correlations)

# Class proportion
class_counts = data.groupby('class').size()
print(class_counts)

# ==================================
# 4. DATA VISUALIZATION
# ==================================

# Histograms
data.hist(figsize=(10,10))
pyplot.show()

# Density Plots
data.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False, figsize=(10,10))
pyplot.show()

# Box and Whisker Plots
data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False, sharey = False, figsize=(10,10))
pyplot.show()

# Correlation Matrix Plot
fig = pyplot.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# ==================================
# 5. DATA PREPROCESSING
# ==================================

# --- Rescale Data (between 0 and 1) ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# Summarize transformed data
set_printoptions(precision=3)
print("Rescaled X (first 5 rows):")
print(rescaledX[0:5,:])

# --- Standardize Data (0 mean, 1 stdev) ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# Summarize transformed data
set_printoptions(precision=3)
print("Standardized X (first 5 rows):")
print(rescaledX[0:5,:])

# ==================================
# 6. FEATURE SELECTION
# ==================================

# --- Feature Extraction with RFE ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression(solver='liblinear', max_iter=200)
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# --- Feature Extraction with PCA ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

# --- Feature Importance with Extra Trees Classifier ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier(random_state=7)
model.fit(X, Y)
print(model.feature_importances_)

# ==================================
# 7. MODEL EVALUATION
# ==================================

# --- Evaluate Algorithms (Cross-Validation) ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# prepare models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', max_iter=200)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=7)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto', random_state=7)))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# --- Compare Algorithms (Boxplot) ---
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results, labels=names)
pyplot.show()

# ==================================
# 8. MODEL IMPROVEMENT
# ==================================

# --- Standardize the data and evaluate Logistic Regression ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# standardize data
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# evaluate algorithms
model = LogisticRegression(solver='liblinear', max_iter=200)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, rescaledX, Y, cv=kfold)
print("LR scaled: %.3f (%.3f)" % (results.mean(), results.std()))

# --- KNN Algorithm tuning ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# standardize data
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# Tune KNN
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
results = []
for n_neighbors in neighbors:
	kfold = KFold(n_splits=10, random_state=7, shuffle=True)
	model = KNeighborsClassifier(n_neighbors=n_neighbors)
	cv_results = cross_val_score(model, rescaledX, Y, cv=kfold, scoring='accuracy')
	results.append(cv_results.mean())
	print("KNN neighbors=%d: %.3f" % (n_neighbors, cv_results.mean()))
# plot scores
pyplot.plot(neighbors, results)
pyplot.xlabel('Number of Neighbors')
pyplot.ylabel('Accuracy')
pyplot.show()

# --- Tune SVM ---
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# standardize data
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# Tune SVM
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC(gamma='auto', random_state=7)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# ==================================
# 9. FINALIZE MODEL
# ==================================

# Finalize Model with cross-validation set aside
# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# Fit the model on 33%
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
