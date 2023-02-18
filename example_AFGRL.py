from data import DatasetAFGRL
from loader import FullLoader

from augment import AugPosDGI, AugNegDGI

from nn.encoders import GCNDGI
from nn.models import ModelAFGRL
from methods import AFGRL
import copy

# data
dataset = DatasetAFGRL()
dataset.load(path="./datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

# neural networks
# TODO: use GCNDGI for now
student_encoder = GCNDGI(dim_in=1433)
teacher_encoder = copy.deepcopy(student_encoder)
model = ModelAFGRL(student_encoder=student_encoder, teacher_encoder=teacher_encoder, adj_ori = dataset.adj_ori_sparse)

# trainer
#TODO: add augmentation
afgrl = AFGRL(model=model, data_loader=data_loader, save_root="./results")
afgrl.train()
