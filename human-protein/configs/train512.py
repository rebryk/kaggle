from attrdict import AttrDict

config = AttrDict()

config.log_path = '/rebryk/kaggle/protein/logs'
config.data_path = '/rebryk/kaggle/protein/dataset'
config.model_path = '/rebryk/kaggle/protein/models'
config.submission_path = '/rebryk/kaggle/protein/submissions'
config.tensorboard_path = '/rebryk/logs'

config.exp = 'stage_2'
config.model = 'resnet50'
config.num_workers = 8
config.batch_size = 32
config.image_size = 512
config.lr = 0.10
config.num_epochs = [6]
config.cycles_len = [2]
config.lr_divs = [8]
config.test_size = 0.1
config.k_fold = 5
config.external_data = True
config.use_sampler = True
config.mixed_precision = False
config.checkpoint = 'stage_1_sz256_x128_f{}_17'
config.n_aug_train = None
config.n_aug_test = None
