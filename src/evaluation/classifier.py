import torch
import torch.nn as nn
from .base_evaluator import Evaluator
import numpy as np
import warnings

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

class Evaluator(nn.Module):
    def forward(self)->None:
        raise NotImplementedError
    
class LogReg(Evaluator):
    def __init__(self, hid_units, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_units, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

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

