from data import DatasetDGI
from loader import FullLoader

from augment import RandomMask, RandomDropEdge, RandomDropNode, AugmentSubgraph

from nn.encoders import GCNDGI
from nn.utils import DiscriminatorDGI
from nn.models import ModelDGI
from methods import GraphCL

# data
dataset = DatasetDGI()
dataset.load(path="./datasets/cora_dgi")
data = dataset.to_data()
data_loader = FullLoader(data)

# neural networks
encoder = GCNDGI(dim_in=1433)
discriminator = DiscriminatorDGI(dim_h=512)
model = ModelDGI(encoder=encoder, discriminator=discriminator)

aug_type = 'subgraph'

# trainer
if aug_type == 'edge':
    augment_neg = RandomDropEdge()
elif aug_type == 'mask':
    augment_neg = RandomMask()
elif aug_type == 'node':
    augment_neg = RandomDropNode()
elif aug_type == 'subgraph':
    augment_neg = AugmentSubgraph()
else:
    assert False

GraphCL = GraphCL(model=model, data_loader=data_loader, data_augment=augment_neg, emb_augment=None,
                  save_root="./results", n_epochs=10)
GraphCL.train()

# embeds = GraphCL.model.get_embs(x_pos, adj, self.is_sparse)

