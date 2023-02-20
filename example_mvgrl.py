from data import DatasetDGI, DatasetMVGRL
from loader import FullLoader
from augment.collections import augment_dgi


from augment import AugPosMVGRL, AugPPRMVGRL, AugHeatMVGRL

from nn.encoders import GCNMVGRL
from nn.utils import DiscriminatorMVGRL
from nn.models import ModelMVGRL
from methods import MVGRL


### MVGRL ###
dataset = DatasetMVGRL()
dataset.load(path="./datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

encoder_1 = GCNMVGRL(dim_in=1433)
encoder_2 = GCNMVGRL(dim_in=1433)
discriminator = DiscriminatorMVGRL(dim_h=512)
model = ModelMVGRL(encoder=[encoder_1,encoder_2], discriminator=discriminator)

augment_pos = AugPosMVGRL()
augment_neg = AugPPRMVGRL()
mvgrl = MVGRL(model=model, data_loader=data_loader, augment_pos=AugPosMVGRL(), augment_neg=AugPPRMVGRL(), save_root="./results")
mvgrl.train()

embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), is_numpy=True)

