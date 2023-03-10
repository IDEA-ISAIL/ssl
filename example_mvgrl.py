from data import DatasetDGI, DatasetMVGRL
from loader import FullLoader
from augment.collections import augment_mvgrl_ppr, augment_mvgrl_heat


# from augment import AugPosMVGRL, AugPPRMVGRL, AugHeatMVGRL

from nn.encoders import GCNDGI
from nn.utils import DiscriminatorMVGRL
from nn.models import ModelMVGRL
from methods import MVGRL


### MVGRL ###
dataset = DatasetMVGRL()
dataset.load(path="./datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

encoder_1 = GCNDGI(dim_in=1433)
encoder_2 = GCNDGI(dim_in=1433)
discriminator = DiscriminatorMVGRL(dim_h=512)
model = ModelMVGRL(encoder=[encoder_1,encoder_2], discriminator=discriminator)

mvgrl = MVGRL(model=model, data_loader=data_loader, data_augment = augment_mvgrl_ppr, n_epochs = 10, sample_size = 500, save_root="./results")
mvgrl.train()

embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), diff = data.adj.cuda())

