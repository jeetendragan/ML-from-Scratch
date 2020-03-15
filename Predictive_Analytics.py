# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""

import pandas as pd
import numpy as np
#import matplotlib.pylot as plot
import random_forest
import random
import time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    x = y_true == y_pred
    acc = x.sum()/len(x)
    return acc

def Recall(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float

    """

    cm = ConfusionMatrix(y_true, y_pred)
    rec = np.diag(cm) / np.sum(cm, axis = 1)
    recall = np.mean(rec)*100
    return recall

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    cm = ConfusionMatrix(y_true,y_pred)
    prec = np.diag(cm) / np.sum(cm, axis = 0)
    precision = np.mean(prec)*100
    return precision
    
def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    
def ConfusionMatrix(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """ 
    num_class = len(np.unique(y_true))
    shift = num_class*y_true.min()
    con_matrix = y_true * num_class + y_pred
    distri = np.histogram(con_matrix,bins = pow(num_class,2),range=(y_true.min()+shift, pow(num_class,2)+shift))
    cm = np.reshape(distri[0], (num_class, num_class))
    return cm

def KNN(X_train,X_test,Y_train):
     """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """

    train = pd.DataFrame(X_train)
    train['48'] = Y_train
    test = pd.DataFrame(X_test)

    geni = random_forest.calculate_geni(data_train, '48')
    node = random_forest.Node(geni, range(0, train.shape[0]))
    forest = random_forest.RandomForest(4, 6, 10, 12)

    start = time.time()
    forest.build_random_forest(data_train, '48')
    end = time.time()
    print("Time taken to build the random forest(minutes): "+str((end - start)/60))

    start = time.time()
    predictions = forest.predict(data_test)
    end = time.time()
    print("Time taken for predictions(minutes): "+str((end - start)/60))

    data_test_rst_index = data_test.reset_index(drop = True)
    accuracy = (predictions == data_test_rst_index['48']).sum()/data_test.shape[0]
    print("The accuracy obtained using random forest is: "+str(accuracy))
    return predictions
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    X_train = X_train - np.mean(X_train, axis = 0)
    cov = np.cov(X_train, rowvar = False)
    val, vec = np.linalg.eigh(cov)
    sorted_eig = np.argsort(val)[::-1]
    vec = vec[:, sorted_eig]
    val = val[sorted_eig]
    vec = vec[:, :N]
    df_vec = pd.DataFrame(vec)
    pca = np.dot(X_train, vec)
    return pca
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """

def SklearnSupervisedLearning(X_train,Y_train, X_test, Y_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    Assumption: X_train and Y_train are not scaled

    :rtype: List[numpy.ndarray] 
    """

    # as X_train, Y_train are not scaled, we will have to scale them as some implementations require scaled data sets
    scaler = MinMaxScaler()
    complete_X = np.concatenate((X_train, X_test), axis=0)
    scaled_data = scaler.fit_transform(complete_X)
    scaled_train_X = scaled_data[0:X_train.shape[0]]
    scaled_test_X = scaled_data[X_train.shape[0]: X_train.shape[0]+X_test.shape[0]]
    
    model_predictions = list()
    # test Sklearn - SVM
    print("Training SVM")
    predictions_svm = sk_SVM(scaled_train_X, Y_train, scaled_test_X)
    acc = Accuracy(Y_test, predictions_svm)
    print("Accuracy(SVM): "+str(acc))
    model_predictions.append(predictions_svm)
    
    # test SkLearn - Logistic
    print("Training Logistic")
    predictions_log = sk_logistic(scaled_train_X, Y_train, scaled_test_X)
    acc = Accuracy(Y_test, predictions_log)
    print("Accuracy(Logistic): "+str(acc))
    model_predictions.append(predictions_log)

    # test SkLearn - decision tree
    print("Training decision tree")
    predictions_dt = sk_decision_tree(X_train, Y_train, X_test)
    acc = Accuracy(Y_test, predictions_dt)
    print("Accuracy(Decison Tree): "+str(acc))
    model_predictions.append(predictions_dt)

    # test SkLearn - KNN
    print("Training KNN")
    predictions_knn = sk_Knn(scaled_train_X, Y_train, scaled_test_X)
    acc = Accuracy(Y_test, predictions_knn)
    print("Accuracy(KNN): "+str(acc))
    model_predictions.append(predictions_knn)

    return model_predictions

def sk_SVM(X_train,Y_train, X_test):
    # Please pass in scaled data - MinMaxNormalized
    svm_classifier = SVC(kernel = 'linear', random_state = 0, C = 10, degree = 3)
    svm_classifier.fit(X_train, Y_train)
    print("Training SVM complete. Predicting....")
    svm_pred = svm_classifier.predict(X_test)
    return svm_pred

def sk_logistic(X_train, Y_train, X_test):
    # important to pass in scaled data
    #Y_train = Y_train.astype('int')
    classifier_log = LogisticRegression(random_state = 0, max_iter=2000)
    classifier_log.fit(X_train, Y_train)
    y_pred_log = classifier_log.predict(X_test)
    return y_pred_log

def sk_decision_tree(X_train, Y_train, X_test):
    # No need to pass scaled data.
    classifier_dt = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
    classifier_dt.fit(X_train, Y_train)
    y_pred_dt = classifier_dt.predict(X_test)
    return y_pred_dt

def sk_Knn(X_train, Y_train, X_test):
    # this method also requires normalized data
    classifier = KNeighborsClassifier(metric = 'euclidean', n_neighbors= 12)
    classifier.fit(X_train, Y_train)
    return classifier.predict(X_test)

def SklearnVotingClassifier(X_train,Y_train,X_test, Y_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    :type Y_test: numpy.ndarray
    Assumption: the X_train, and X_test are not scaled
    
    :rtype: List[numpy.ndarray] 
    """
    predictions = SklearnSupervisedLearning(X_train, Y_train, X_test, Y_test)
    predf = pd.DataFrame(predictions)
    pred_mode = predf.mode()
    top_row = pred_mode.iloc[0:1, :]
    final_preds = top_row.to_numpy()[0]

    acc = Accuracy(Y_test, final_preds)
    print("Ensemble accuracy: "+str(acc))

    return final_preds

def SVMGridSearch(train, response):
    C = [10, 1000, 5000]
    degree = [1, 2, 3]
    parameters = [{'C': C, 'kernel': ['linear'], 'degree': degree}]
    classifier = SVC(kernel = 'linear', random_state = 0)
    grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 2,
                            n_jobs = -1)
    grid_search = grid_search.fit(train.drop(columns = [response]), train[response])
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    cv_result = pd.DataFrame(grid_search.cv_results_)
    fig = plt.figure()
    ax = plt.subplot(111)
    counter = 0
    for deg in degree:
        st = 'Degree = '+str(deg)
        ax.plot(C, cv_result['mean_test_score'][counter: counter+len(degree)], label=st)
        counter = counter+len(degree)
    
    plt.title("Parameter tuning for SVM")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    ax.legend()
    plt.show()

def DecisionTreeGridSearch(train, response):
    classifier = DecisionTreeClassifier(random_state=0)
    print("Performing grid search on parameters: max_features, and criterion.\n")
    parameters = [{'max_features': [1, 3, 5, 7, 9, 12, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33], 'criterion': ['gini','entropy']}]
    grid_search_ent = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 5,
                            n_jobs = -1)
    grid_search_ent = grid_search_ent.fit(train.drop(columns = ['48']), train['48'])
    best_accuracy = grid_search_ent.best_score_
    best_parameters = grid_search_ent.best_params_
    cv_result = pd.DataFrame(grid_search_ent.cv_results_)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(parameters[0]['max_features'], cv_result['mean_test_score'][0:len(parameters[0]['max_features'])], label='Geni')
    ax.plot(parameters[0]['max_features'], cv_result['mean_test_score'][len(parameters[0]['max_features']):len(cv_result)], label = 'Entropy')
    plt.title("Parameter tuning for decision tree")
    plt.xlabel("max_features")
    plt.ylabel("Accuracy")
    ax.legend()
    plt.show()

def KnnGridSearch(X_train, y_train, response):
    # normalize the data
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)

    classifier = KNeighborsClassifier(metric = 'euclidean')
    print("Performing grid search on parameters: Knearest neighbours.")
    parameters = [{'n_neighbors': [5, 8 ,10, 12, 25]}]
    grid_search_knn = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 2,
                            n_jobs = -1)
    grid_search_knn = grid_search_knn.fit(X_train, y_train)
    best_accuracy = grid_search_knn.best_score_
    best_parameters = grid_search_knn.best_params_
    cv_result = pd.DataFrame(grid_search_knn.cv_results_)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(parameters[0]['n_neighbors'], cv_result['mean_test_score'][0:len(parameters[0]['n_neighbors'])], label="Knn")
    plt.title("Parameter tuning for KNN")
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    ax.legend()
    plt.show()

"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    train_size = int(data.shape[0]*0.7)
    indices = random.sample(range(0, data.shape[0]), train_size)
    data_train = data.iloc[indices]
    index_mask = data.index.isin(indices)
    data_test = data[~index_mask]

    scaler = MinMaxScaler()
    scaled_data_X = pd.DataFrame(scaler.fit_transform(data.drop(columns = ['48'])))
    scaled_train_data_X = scaled_data_X[index_mask]
    train_y = data['48'][index_mask]
    scaled_test_data_X = scaled_data_X[~index_mask]
    test_y = data['48'][~index_mask]

    ## testing the functions ##

    #predictions = RandomForest(data_train.drop(columns= ['48']), data_train['48'], data_test)
    #DecisionTreeGridSearch(data_train, '48')

    some_train = data_train.iloc[0:100]
    #SVMGridSearch(some_train, '48')
    #KnnGridSearch(data.drop(columns = ['48']).to_numpy(), data['48'].to_numpy(), '48')

    # test Sklearn - SVM
    #predictions = sk_SVM(data_train.drop(columns = ['48']).to_numpy(), data_train['48'].to_numpy(), data_test.drop(columns = ['48']).to_numpy())

    # test SkLearn - Logistic
    #predictions = sk_logistic(scaled_train_data_X.to_numpy(), train_y.to_numpy(), scaled_test_data_X.to_numpy())

    # test SkLearn - decision tree
    #predictions = sk_decision_tree(data_train.drop(columns = ['48']), data_train['48'], data_test.drop(columns = ['48']))

    # test SkLearn - KNN
    #predictions = sk_logistic(scaled_train_data_X.to_numpy(), train_y.to_numpy(), scaled_test_data_X.to_numpy())
    #acc = Accuracy(data_test['48'], predictions)
    #print("Acc: "+str(acc))

    # test SklearnSupervisedLearning
    # predictions = SklearnSupervisedLearning(data_train.drop(columns = ['48']).to_numpy(), data_train['48'].to_numpy(), data_test.drop(columns = ['48']).to_numpy(), data_test['48'].to_numpy())

    # test SklearnVotingClassifier
    predictions = SklearnVotingClassifier(data_train.drop(columns = ['48']).to_numpy(), data_train['48'].to_numpy(), data_test.drop(columns = ['48']).to_numpy(), data_test['48'].to_numpy())