from data import DatasetDGI, DatasetMVGRL
from loader import FullLoader
from augment.collections import augment_dgi


#from augment import AugPosDGI, AugNegDGI, AugPosMVGRL, AugPPRMVGRL, AugHeatMVGRL
#
#from nn.encoders import GCNDGI, GCNMVGRL
#from nn.utils import DiscriminatorDGI, DiscriminatorMVGRL
#from nn.models import ModelDGI, ModelMVGRL
#from methods import DGI, MVGRL
#
from nn.utils import DiscriminatorDGI
from nn.encoders import GCNDGI
from nn.models import ModelDGI
from methods import DGI

from evaluation import eval


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

### MVGRL ###
# dataset = DatasetMVGRL()
# dataset.load(path="./datasets/cora_dgi")
# data = dataset.to_data()
# data_loader = FullLoader(data)

# encoder_1 = GCNMVGRL(dim_in=1433)
# encoder_2 = GCNMVGRL(dim_in=1433)
# discriminator = DiscriminatorMVGRL(dim_h=512)
# model = ModelMVGRL(encoder=[encoder_1,encoder_2], discriminator=discriminator)

# augment_pos = AugPosMVGRL()
# augment_neg = AugPPRMVGRL()
# mvgrl = MVGRL(model=model, data_loader=data_loader, augment_pos=AugPosMVGRL(), augment_neg=AugPPRMVGRL(), save_root="./results")
# mvgrl.train()

#evaluator
embs = model.get_embs(x=data.x.cuda(), adj=data.adj.cuda(), is_numpy=True)
labels = dataset.labels

# node classification/ link prediction/ graph reconstruction

eval(embs, labels, evaluator = "similarity_search")



