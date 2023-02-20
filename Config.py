import yaml


class model_config:
    def __init__(self, entries):
        self.__dict__.update(entries)
        if not hasattr(self, 'dropout'):
            self.dropout = 0
        if not hasattr(self, 'n_layers'):
            self.n_layers = 2
        if not hasattr(self, 'd_hidden'):
            self.d_hidden = 100
        if not hasattr(self, 'backbone'):
            self.backbone = 'gcn'
        if not hasattr(self, 'head') and self.backbone == 'gat':
            self.head = 1
        if not hasattr(self, 'normalize'):
            self.normalize = False


class optimizer_config:
    def __init__(self, entries):
        self.__dict__.update(entries)
        if not hasattr(self, 'lr'):
            self.lr = 5e-4
        if not hasattr(self, 'name'):
            self.name = 'adam'
        if not hasattr(self, 'epoch'):
            self.epoch = 3000
        if not hasattr(self, 'patience'):
            self.patience = 1000
        if not hasattr(self, 'use_gpu'):
            self.use_gpu = False
        if self.use_gpu and not hasattr(self, 'gpu_idx'):
            self.gpu_idx = 0
        if not hasattr(self, 'weight_decay'):
            self.weight_decay = 5e-4


class dataset_config:
    def __init__(self, entries):
        self.__dict__.update(entries)
        if not hasattr(self, 'name'):
            self.name = 'cora'
        if not hasattr(self, 'data_dir'):
            self.data_dir = 'ssl/data/cora/preprocessed_data.mat'


class output_config:
    def __init__(self, entries):
        self.__dict__.update(entries)
        if not hasattr(self, 'verbose'):
            self.verbose = True
        # if not hasattr(self, 'save_model'):
        #     self.save_model = True
        if not hasattr(self, 'save_dir'):
            self.save_dir = './saved_model'
        if self.verbose and not hasattr(self, 'interval'):
            self.interval = 100


class Config:
    def __init__(self, configuration):
        for items in configuration:
            if items == 'model':
                self.model = model_config(configuration[items])
            elif items == 'optim':
                self.optim = optimizer_config(configuration[items])
            elif items == 'dataset':
                self.dataset = dataset_config(configuration[items])
            elif items == 'output':
                self.output = output_config(configuration[items])
            else:
                raise ValueError('{} is not an object in Config'.format(items))

    # def print(self):


def load_config(dir):
    with open(dir, 'r') as yamlfile:
        configuration = yaml.safe_load(yamlfile)
    config = Config(configuration)
    return config


