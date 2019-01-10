from attrdict import AttrDict

config = AttrDict()

config.log_path = '/rebryk/kaggle/protein/logs'
config.data_path = '/rebryk/kaggle/protein/dataset'
config.model_path = '/rebryk/kaggle/protein/models'
config.submission_path = '/rebryk/kaggle/protein/submissions'
config.tensorboard_path = '/rebryk/logs'

config.exp = 'test'
config.model = 'resnet50'
config.num_workers = 8
config.batch_size = 32
config.image_size = 512
config.lr = None
config.num_epochs = None
config.cycles_len = None
config.lr_divs = None
config.test_size = 0.1
config.k_fold = 5
config.external_data = False
config.use_sampler = True
config.mixed_precision = False
config.checkpoint = 'stage_2_sz512_x32_f{}_06'
config.n_aug_train = 0
config.n_aug_test = 8
