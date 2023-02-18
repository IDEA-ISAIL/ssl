from data import DatasetDGI
from loader import FullLoader

from augment import AugPosDGI, AugNegDGI

from nn.encoders import GCNDGI
from nn.models import ModelBGRL
from methods import BGRL
import copy

# data
dataset = DatasetDGI()
dataset.load(path="./datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

# neural networks
# TODO: use GCNDGI for now
student_encoder = GCNDGI(dim_in=1433)
teacher_encoder = copy.deepcopy(student_encoder)
model = ModelBGRL(student_encoder=student_encoder, teacher_encoder=teacher_encoder)

# trainer
#TODO: add augmentation
bgrl = BGRL(model=model, data_loader=data_loader, save_root="./results")
bgrl.train()
