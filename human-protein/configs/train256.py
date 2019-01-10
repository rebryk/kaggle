from attrdict import AttrDict

config = AttrDict()

config.log_path = '/rebryk/kaggle/protein/logs'
config.data_path = '/rebryk/kaggle/protein/dataset'
config.model_path = '/rebryk/kaggle/protein/models'
config.submission_path = '/rebryk/kaggle/protein/submissions'
config.tensorboard_path = '/rebryk/logs'

config.exp = 'stage_1'
config.model = 'resnet50'
config.num_workers = 8
config.batch_size = 128
config.image_size = 256
config.lr = 0.10
config.num_epochs = [1, 8, 8]
config.cycles_len = [0, 2, 4]
config.lr_divs = [0, 4, 12]
config.test_size = 0.1
config.k_fold = 5
config.external_data = True
config.use_sampler = True
config.mixed_precision = False
config.checkpoint = None
config.n_aug_train = None
config.n_aug_test = None
