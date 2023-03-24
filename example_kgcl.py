from data.dataset_kgcl import *
from loader import FullLoader
from augment.collections import augment_dgi
from os.path import join
import multiprocessing
import argparse


from augment.kgcl import AugKGCL

from nn.encoders import GCNMVGRL
from nn.utils import DiscriminatorMVGRL
from nn.models.kgcl import ModelKGCL
from methods.kgcl import KGCL


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.8,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=4096,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='yelp2018',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='kgcl', help='rec-model, support [mf, lgn, kgcl]')
    return parser.parse_args()



class Config:

    def __init__(self, args):
        self.ROOT_PATH = "./"
        self.CODE_PATH = join(self.ROOT_PATH, 'code')
        self.DATA_PATH = join(self.ROOT_PATH, 'datasets')
        self.BOARD_PATH = join(self.CODE_PATH, 'runs')
        self.FILE_PATH = join(self.CODE_PATH, 'checkpoints')
        import sys
        sys.path.append(join(self.CODE_PATH, 'sources'))


        if not os.path.exists(self.FILE_PATH):
            os.makedirs(self.FILE_PATH, exist_ok=True)


        self.config = {}
        self.all_dataset = ['movielens', 'last-fm', 'MIND', 'yelp2018', 'amazon-book']
        self.all_models  = ['lgn','kgcl','sgl']
        # config['batch_size'] = 4096
        self.config['bpr_batch_size'] = args.bpr_batch
        self.config['latent_dim_rec'] = args.recdim
        self.config['lightGCN_n_layers']= args.layer
        self.config['dropout'] = args.dropout
        self.config['keep_prob']  = args.keepprob
        self.config['A_n_fold'] = args.a_fold
        self.config['test_u_batch_size'] = args.testbatch
        self.config['multicore'] = args.multicore
        self.config['lr'] = args.lr
        self.config['decay'] = args.decay
        self.config['pretrain'] = args.pretrain
        self.config['A_split'] = False

        self.GPU = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU else "cpu")

        self.kgcn = "RGAT"
        self.train_trans = True
        self.entity_num_per_item = 10
        # WEIGHTED (-MIX) \ RANDOM \ ITEM-BI \ PGRACE \NO
        self.uicontrast = "RANDOM"
        self.kgc_enable = True
        self.kgc_joint = True
        self.kgc_temp = 0.2
        self.use_kgc_pretrain = False
        self.pretrain_kgc = False
        self.kg_p_drop = 0.5
        self.ui_p_drop = 0.001
        self.ssl_reg = 0.1
        self.CORES = multiprocessing.cpu_count() // 2
        self.seed = args.seed

        self.test_verbose = 1
        self.test_start_epoch = 1
        self.early_stop_cnt = 10

        self.dataset = args.dataset
        if self.dataset=='MIND':
            self.config['lr'] = 5e-4
            self.config['decay'] = 1e-3
            self.config['dropout'] = 1
            self.config['keep_prob']  = 0.6

            self.uicontrast = "WEIGHTED-MIX"
            self.kgc_enable = True
            self.kgc_joint = True
            self.use_kgc_pretrain = False
            self.entity_num_per_item = 6
            # [0.06, 0.08, 0.1]
            self.ssl_reg = 0.06
            self.kgc_temp = 0.2
            # [0.3, 0.5, 0.7]
            self.kg_p_drop = 0.5
            # [0.1, 0.2, 0.4]
            self.ui_p_drop = 0.4
            self.mix_ratio = 1-self.ui_p_drop-0
            self.test_start_epoch = 1
            self.early_stop_cnt = 3
            
        elif self.dataset=='amazon-book':
            self.config['dropout'] = 1
            self.config['keep_prob']  = 0.8
            self.uicontrast = "WEIGHTED"
            self.ui_p_drop = 0.05
            self.mix_ratio = 0.75
            self.test_start_epoch = 15
            self.early_stop_cnt = 5

        elif self.dataset=='yelp2018':
            self.config['dropout'] = 1
            self.config['keep_prob']  = 0.8
            self.uicontrast = "WEIGHTED"
            self.ui_p_drop = 0.1
            self.test_start_epoch = 25
            self.early_stop_cnt = 5


        self.model_name = args.model
        if self.dataset not in self.all_dataset:
            raise NotImplementedError(f"Haven't supported {dataset} yet!, try {self.all_dataset}")
        if self.model_name not in self.all_models:
            raise NotImplementedError(f"Haven't supported {self.model_name} yet!, try {self.all_models}")
        if self.model_name == 'lgn':
            self.kgcn = "NO"
            self.train_trans = False
            # WEIGHTED \ RANDOM \ ITEM-BI \ PGRACE \NO
            self.uicontrast = "NO"
            self.kgc_enable = False
            self.kgc_joint = False
            self.use_kgc_pretrain = False
            self.pretrain_kgc = False
        elif self.model_name == 'sgl':
            self.kgcn = "NO"
            self.train_trans = False
            # WEIGHTED \ RANDOM \ ITEM-BI \ PGRACE \NO
            self.uicontrast = "RANDOM"
            self.kgc_enable = False
            self.kgc_joint = False
            self.use_kgc_pretrain = False
            self.pretrain_kgc = False



        self.TRAIN_epochs = args.epochs
        self.LOAD = args.load
        self.PATH = args.path
        self.topks = eval(args.topks)
        self.tensorboard = args.tensorboard
        self.comment = args.comment
        # let pandas shut up
        from warnings import simplefilter
        simplefilter(action="ignore", category=FutureWarning)





### MVGRL ###
args = parse_args()

config = Config(args)
kg_dataset = KGDatasetKGCL(config)
kg_dataset.load(path="./datasets/yelp2018/kg.txt")
path=join(config.DATA_PATH, config.dataset)
dataset = PurchaseDatasetKGCL(config.config, path, config)

Recmodel = ModelKGCL(config, dataset, kg_dataset)
Recmodel = Recmodel.to(config.device)


augment_pos = AugKGCL(Recmodel, config, config.kgc_temp)

kgcl = KGCL(model=Recmodel, config=config, dataset=dataset, kg_dataset=kg_dataset, augment_pos=augment_pos)
kgcl.train()

