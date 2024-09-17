from src.augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph, AugmentorList
from src.methods import GCN, Merit
from src.trainer import SimpleTrainer
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from src.transforms import NormalizeFeatures, GCNNorm, Edge2Adj, Compose
from src.datasets import Planetoid, Amazon, WikiCS,Coauthor
from src.utils.create_data import create_masks
from src.evaluation import LogisticRegression
import torch 
import yaml
from src.utils.add_adj import add_adj_t
from sklearn.impute import SimpleImputer



torch.manual_seed(0)
config = yaml.safe_load(open("./configuration/merit.yml", 'r', encoding='utf-8').read())
# data
pre_transforms = Compose([NormalizeFeatures(ord=1), Edge2Adj(norm=GCNNorm(add_self_loops=1))])
data_name = config['dataset']

if data_name=="photo":
    dataset = Amazon(root="pyg_data", name="photo", pre_transform=pre_transforms) 
elif data_name=="coauthor": 
    dataset = Coauthor(root="pyg_data", name='cs', transform=pre_transforms)
elif data_name=="wikics": 
    dataset = WikiCS(root="pyg_data", transform=T.NormalizeFeatures())
    dataset = add_adj_t(dataset)
    nan_mask = torch.isnan(dataset[0].x)
    imputer = SimpleImputer()
    dataset[0].x = torch.tensor(imputer.fit_transform(dataset[0].x))
    
data_loader = DataLoader(dataset)
data = dataset.data

device = 'cuda:0'

def main():

  # ------------------- Method -----------------
  encoder = GCN(in_ft=data.x.shape[1], out_ft=512, projection_hidden_size=config["projection_hidden_size"],
                    projection_size=config["projection_size"])
  
  # model = GCNLayer(data.x.shape[1], 512)
  # method = MERIT(gnn=model,
  #               feat_size=data.x.shape[1],
  #               projection_size=512,
  #               projection_hidden_size=4096,
  #               prediction_size=512,
  #               prediction_hidden_size=4096,
  #               moving_average_decay=0.8, beta=0.6).to(device)

  # opt = torch.optim.Adam(merit.parameters(), lr=lr, weight_decay=weight_decay)
  
  method = Merit(encoder=encoder, data = data, config=config,device=device,is_sparse=True)
  
  # method.augment_type = aug_type

  # print(method)

  # ------------------ Trainer --------------------
  # trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device)
  trainer = SimpleTrainer(method=method, data_loader=data_loader, device=device, n_epochs=config['epochs'], lr=config['lr'])
  
  # return 
  
  trainer.train()


  # ------------------ Evaluator -------------------
  data_pyg = dataset.data.to(method.device)
  embs = method.get_embs(data_pyg, data_pyg.adj_t).detach()

  lg = LogisticRegression(lr=0.01, weight_decay=0, max_iter=2000, n_run=50, device=device)
  create_masks(data=data_pyg.cpu())
  lg(embs=embs, dataset=data_pyg)

main()