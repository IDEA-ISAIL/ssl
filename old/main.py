import argparse
import numpy as np
import time
import torch
from module import GCN, GraphSAGE, GAT
from ogb.nodeproppred import Evaluator
import torch.nn.functional as F
from Config import load_config
from Dataloader import load_graph


def main(config, data, idx):
    use_cuda = config.optim.use_gpu and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:{}'.format(config.optim.gpu_idx))
    else:
        device = torch.device('cpu')
    data.num_classes = torch.unique(data.y).shape[0]
    data.x, data.y = data.x.to(device), data.y.to(device)
    data.adj_t = data.adj_t.to_symmetric()
    data.adj_t = data.adj_t.to(device)
    meters = {'acc': 0, 'iter': 0, 'best_y': None}
    if config.model.backbone == 'GCN':
        model = GCN(data.num_node_features, config.model.d_hidden, data.num_classes, config.model.dropout,
                    config.model.n_layers, config.model.normalize).float()
    elif config.model.backbone == 'GAT':
        model = GAT(data.num_node_features, config.model.d_hidden, data.num_classes, config.model.dropout,
                    config.model.n_layers, 5, config.model.normalize).float()
    else:
        model = GraphSAGE(data.num_node_features, config.model.d_hidden, data.num_classes, config.model.dropout,
                          config.model.n_layers, config.model.normalize).float()
    if use_cuda:
        model.to(device)
    if config.optim.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.name == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.name == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    print("start training...")
    model.train()
    t0 = time.time()
    criterion = F.nll_loss
    for epoch in range(1, config.optim.epoch+1):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_idx], data.y[data.train_idx])
        loss.backward()
        optimizer.step()
        train_acc, valid_acc, test_acc = test(model, data, meters, config, out=out, idx=idx, eval=False)
        if config.output.verbose and epoch % config.output.interval == 0:
            print("Epoch {:05d} | Time(s) {:.4f} | Train Accuracy: {:.4f} | Val Accuracy: {:.4f} | Test Accuracy: {:.4f}"
                  " | Loss: {:.4f}".format(epoch, time.time()-t0, train_acc, valid_acc, test_acc, loss))
            t0 = time.time()
        model.train()
        if meters['iter'] >= config.optim.patience:
            break
        meters['iter'] += 1
    model.load_state_dict(torch.load('{}/best_{}_model_{}.pt'.format(config.output.save_dir,
                                                                      config.model.backbone, idx)).state_dict())
    train_acc, valid_acc, test_acc = test(model, data, meters, config, idx=idx, eval=True)
    print("====>Final Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(
        train_acc, valid_acc, test_acc))
    return [train_acc, valid_acc, test_acc]


@torch.no_grad()
def test(model, data, meters, config, out=None, idx=0, eval=True):
    evaluator = Evaluator(name='ogbn-arxiv')
    model.eval()
    if out is None:
        out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': data.y[data.train_idx].reshape(-1, 1),
        'y_pred': y_pred[data.train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_idx].reshape(-1, 1),
        'y_pred': y_pred[data.val_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_idx].reshape(-1, 1),
        'y_pred': y_pred[data.test_idx],
    })['acc']
    if valid_acc >= meters['acc'] and not eval:
        meters['acc'] = valid_acc
        torch.save(model, '{}/best_{}_model_{}.pt'.format(config.output.save_dir, config.model.backbone, idx))
        meters['iter'] = 0
        meters['best_y'] = out
    return train_acc, valid_acc, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deeper-GXX')
    # parser.add_argument("-d_data", type=str, default='./data/cora', help="directory of the dataset")
    parser.add_argument("-d_config", type=str, default='./config/cora', help="directory of the configuration file")
    args = parser.parse_args()
    config = load_config(args.d_config)
    results = []
    for i in range(5):
        print('Round {}'.format(i+1))
        data = load_graph(config.dataset, index=i, p=0)
        np.random.seed(i+1)
        torch.manual_seed(i+1)
        results.append(main(config, data, i))
    results = np.array(results)
    avg_acc = results.mean(axis=0)
    std_acc = results.std(axis=0)
    print('Overall Performance:')
    print('Train Accuracy: {:.4f}, +/- {:.4f}'.format(avg_acc[0], std_acc[0]))
    print('Valid Accuracy: {:.4f}, +/- {:.4f}'.format(avg_acc[1], std_acc[1]))
    print('Test Accuracy: {:.4f},  +/- {:.4f}'.format(avg_acc[2], std_acc[2]))

