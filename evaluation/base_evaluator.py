import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn import preprocessing


class Evaluator(nn.Module):
    def forward(self)->None:
        raise NotImplementedError

from .classifier import logistic_classify, svc_classify, linearsvc_classify, randomforest_classify
from .sim_search import run_similarity_search
from .cluster import run_kmeans

def eval(embeds, labels, evaluator = "logistic_classify", search=True, k=10):
    # labels = labels[np.newaxis]
    labels = np.argmax(labels, axis=1)
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeds[0]), np.array(labels)

    acc = 0
    if evaluator == "logistic_classify":
  
        _acc = logistic_classify(x, y)
        if _acc > acc:
            acc = _acc
  
    elif evaluator == "svc_classify":
        _acc = svc_classify(x,y, search)
        if _acc > acc:
            acc = _acc

    elif evaluator == "linearsvc_classify":
        _acc = linearsvc_classify(x, y, search)
        if _acc > acc:
            acc = _acc
   
    elif evaluator == "randomforest_classify":
        _acc = randomforest_classify(x, y, search)
        if _acc > acc:
            acc = _acc
    
    elif evaluator == "similarity_search":
        run_similarity_search(x, y)
        return
    
    elif evaluator == "KMeans":
        run_kmeans(x, y, k)
   
    print('Average accuracy:', acc)
    return acc