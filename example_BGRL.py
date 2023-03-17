from src.data import DatasetDGI
from src.loader import FullLoader
from src.augment.collections import augment_bgrl_1, augment_bgrl_2

from src.nn.encoders import GCNDGI
from src.nn.models.bgrl import Model
from src.methods import BGRL
import copy

# data
dataset = DatasetDGI()
dataset.load(path="datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

# neural networks
# TODO: use GCNDGI for now
student_encoder = GCNDGI(dim_in=1433)
teacher_encoder = copy.deepcopy(student_encoder)
model = Model(student_encoder=student_encoder, teacher_encoder=teacher_encoder)

# trainer
#TODO: add augmentation
bgrl = BGRL(model=model, data_loader=data_loader, save_root="./results",  data_augment_1 = augment_bgrl_1, data_augment_2 = augment_bgrl_2)
bgrl.train()
