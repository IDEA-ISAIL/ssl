import torch
import torch.nn as nn
from .base_evaluator import LogReg
import numpy as np
import warnings

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


def logistic_classify(embeds, labels):
    nb_classes = np.unique(labels).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = embeds.shape[1]

    accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(embeds, labels):

        # test
        train_embs, test_embs = embeds[train_index], embeds[test_index]
        train_lbls, test_lbls= labels[train_index], labels[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())

    return np.mean(accs)

def svc_classify(embeds, labels, search = False):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    accs = []
    for train_index, test_index in kf.split(embeds, labels):

        # test
        x_train, x_test = embeds[train_index], embeds[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accs.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accs)

def randomforest_classify(embeds, labels, search = False):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    accs = []
    for train_index, test_index in kf.split(embeds, labels):

        # test
        x_train, x_test = embeds[train_index], embeds[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accs.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accs)

def linearsvc_classify(embeds, labels, search = False):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    accs = []
    for train_index, test_index in kf.split(embeds, labels):

        # test
        x_train, x_test = embeds[train_index], embeds[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accs.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accs)

def eval(embeds, labels, evaluator = "logistic_classify", search=True):
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
   
    print('Average accuracy:', acc)
    return acc
