from src.data import DatasetAFGRL
from src.loader import FullLoader
from src.augment.collections import augment_afgrl

from src.nn.encoders import GCNDGI
from src.nn.models.afgrl import Model
from src.methods import AFGRL
import copy

# data
dataset = DatasetAFGRL()
dataset.load(path="datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

# neural networks
# TODO: use GCNDGI for now
student_encoder = GCNDGI(dim_in=1433)
teacher_encoder = copy.deepcopy(student_encoder)
# afgrl aug lies in the forward process of the model
model = Model(student_encoder=student_encoder, teacher_encoder=teacher_encoder, data_augment = augment_afgrl, adj_ori = dataset.adj_ori_sparse)

# trainer
#TODO: add augmentation
afgrl = AFGRL(model=model, data_loader=data_loader, save_root="./results")
afgrl.train()
