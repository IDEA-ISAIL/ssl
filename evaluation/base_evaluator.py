import torch
import torch.nn as nn
from .logreg import LogReg
import numpy as np

class Evaluator(nn.Module):
    def forward(self)->None:
        raise NotImplementedError
        

def eval(embeds, dataset):
    labels = torch.FloatTensor(dataset.labels[np.newaxis]).cuda()
    idx_train = torch.LongTensor(dataset.idx_train).cuda()
    idx_val = torch.LongTensor(dataset.idx_val).cuda()
    idx_test = torch.LongTensor(dataset.idx_test).cuda()
    train_lbls = torch.argmax(labels[0, idx_train], dim=1).cuda()
    val_lbls = torch.argmax(labels[0, idx_val], dim=1).cuda()
    test_lbls = torch.argmax(labels[0, idx_test], dim=1).cuda()
    train_embs = embeds[0, idx_train].cuda()
    val_embs = embeds[0, idx_val].cuda()
    test_embs = embeds[0, idx_test].cuda()
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    tot = torch.zeros(1)
    tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(hid_units = 512, nb_classes=labels.shape[1])
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
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
        accs.append(acc * 100)
        print(acc)
        tot += acc

    print('Average accuracy:', tot / 50)
    return tot / 50