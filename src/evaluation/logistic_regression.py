import torch
import torch.nn as nn

import numpy as np

from .base import BaseEvaluator


class LogisticRegression(BaseEvaluator):
    def __init__(self, lr=0.01, weight_decay=0):
        self.lr = lr
        self.weight_decay = weight_decay

    def __call__(self, embs, dataset, name):
        labels = dataset.y
        emb_dim, num_class = embs.shape[1], dataset.y.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):

            train_mask = dataset.train_mask[i]
            dev_mask = dataset.val_mask[i]
            if name == "wikics":
                test_mask = dataset.test_mask
            else:
                test_mask = dataset.test_mask[i]

            classifier = LogisticRegressionClassifier(emb_dim, num_class)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(embs[train_mask], labels[train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(embs[dev_mask], labels[dev_mask])
            test_logits, _ = classifier(embs[test_mask], labels[test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            dev_acc = (torch.sum(dev_preds == labels[dev_mask]).float() /
                       labels[dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == labels[test_mask]).float() /
                        labels[test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('Evaluate node classification results')
        print('** Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(dev_acc, dev_std, test_acc, test_std))


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss
