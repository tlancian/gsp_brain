import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def open_file(file):
    mat = np.array(pd.read_csv(file, header = None, sep = ","))
    
    threshold = np.percentile(mat[np.tril_indices_from(mat, k=-1)], 80)
    
    down = mat<=threshold
    
    mat[down] = 0


    return mat

    
    
def metrics(file):
    
    features = []
    
    net = open_file(file)
    np.fill_diagonal(net, 0)    
    net = nx.from_numpy_array(net, create_using = nx.Graph())
    
    
    dg = dict(net.degree(weight="weight"))
    bc = nx.betweenness_centrality(net, weight = "weight")
    cc = nx.clustering(net, weight="weight")

    for node in range(116):
        features.append(dg[node])
        
    for node in range(116):
        features.append(round(bc[node],3))
    
    for node in range(116):
        features.append(round(cc[node],3))
    
    return features

def reconstructer(signal, s, U, k):
    
    
    diag_d = []
    diag_d_bar = []
    for i in range(len(signal)):
        
        if i in s:
            diag_d.append(1)
            diag_d_bar.append(0)
        else:
            diag_d.append(0)
            diag_d_bar.append(1)
  
    
    D = np.diag(diag_d)
    D_bar = np.diag(diag_d_bar)
    B = np.dot(np.dot(U,D), np.transpose(U))
    

    r = np.dot(D,signal)
    
    DB = np.dot(D_bar, B)
    
    final_r = r + np.dot(DB,r)
    
    for _ in range(k-1):
        final_r = r + np.dot(DB,final_r)
        
    return final_r




    
    

def classificator(data, features):
    
    X = data.filter(regex = "|".join(features))
    y = data.iloc[:, -1]
        
    
    
    acc = []
    sen = []
    spec = []
    
    for _ in range(5):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        estimator = SVC(kernel="linear")
        selector = RFECV(estimator, step=50, min_features_to_select = 2, cv=10, scoring='f1')
        selector = selector.fit(X_train, y_train)

    
        y_pred = selector.predict(X_test)
        
        acc.append(accuracy_score(y_pred, y_test))
        sen.append(precision_score(y_pred, y_test, pos_label = 1))
        spec.append(recall_score(y_pred, y_test, pos_label = 1))
        
    
    return([[round(np.mean(acc),2), round(np.std(acc),2)], [round(np.mean(sen),2), round(np.std(sen),2)], [round(np.mean(spec),2), round(np.std(spec),2)]])
    
    

