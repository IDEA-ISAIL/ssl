from data import DatasetDGI
from loader import FullLoader
from augment.collections import augment_dgi

from nn.utils import DiscriminatorDGI
from nn.encoders import GCNDGI
from nn.models import ModelDGI
from methods import DGI


# data
dataset = DatasetDGI()
dataset.load(path="datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

# neural networks
encoder = GCNDGI(dim_in=1433)
discriminator = DiscriminatorDGI(dim_h=512)
model = ModelDGI(encoder=encoder, discriminator=discriminator)
# model = ModelDGI(encoder=encoder)

# trainer
dgi = DGI(model=model, data_loader=data_loader, data_augment=augment_dgi, save_root="./results")
dgi.train()
