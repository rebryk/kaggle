from typing import Dict, List

import torch
from apex.fp16_utils import network_to_half
from attrdict import AttrDict
from ignite.engine import create_supervised_evaluator, Engine, Events
from ignite.metrics import Metric
from tqdm import tqdm

from commons.ignite import create_supervised_trainer
from commons.loss import ExponentialMovingAverage
from commons.utils import Logger, get_logger


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 metrics: Dict[str, Metric],
                 data_loaders: AttrDict,
                 max_norm: float = None,
                 norm_type: int = 2,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 is_iteration_scheduler: bool = False,
                 device: torch.cuda.device = None,
                 mixed_precision: bool = False,
                 backup_path: str = None,
                 name: str = 'trainer',
                 logger: Logger = None,
                 finished_epochs: int = 0):
        if mixed_precision:
            model = network_to_half(model)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.data_loaders = data_loaders
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scheduler = scheduler
        self.is_iteration_scheduler = is_iteration_scheduler
        self.device = device
        self.backup_path = backup_path
        self.finished_epochs = finished_epochs
        self.finished_iterations = finished_epochs * len(data_loaders.train)

        self.name = name
        self.logger = logger or get_logger()
        self._progress_bar = None
        self._description = 'ITERATION - loss: {:.3f}'

        self._trainer = create_supervised_trainer(model=model,
                                                  optimizer=optimizer,
                                                  loss_fn=criterion,
                                                  max_norm=max_norm,
                                                  norm_type=norm_type,
                                                  device=device,
                                                  mixed_precision=mixed_precision)
        self._register_handlers(self._trainer)
        self._evaluator = create_supervised_evaluator(model, metrics, device)
        self._epoch = 0
        self._iteration = 0
        self._train_loss = ExponentialMovingAverage()

    def _iteration_started(self, engine: Engine):
        self._iteration = engine.state.iteration

        if self.scheduler is not None and self.is_iteration_scheduler:
            self.scheduler.step()
            iteration = self._iteration + self.finished_iterations
            self.logger.scalar_summary(f'train_{self.name}', 'lr', self.scheduler.get_lr()[0], iteration)

    def _iteration_completed(self, engine: Engine):
        self._train_loss.update(engine.state.output)
        train_loss = self._train_loss.value()

        iteration = self._iteration + self.finished_iterations
        self.logger.scalar_summary(f'train_{self.name}', 'iter_loss', train_loss, iteration)

        self._progress_bar.desc = self._description.format(train_loss)
        self._progress_bar.update(1)

    def _epoch_started(self, engine: Engine):
        self._progress_bar = tqdm(initial=0,
                                  leave=False,
                                  total=len(self.data_loaders.train),
                                  desc=self._description.format(0))

        self._epoch = engine.state.epoch

        if self.scheduler is not None and not self.is_iteration_scheduler:
            self.scheduler.step()
            epoch = self._epoch + self.finished_epochs
            self.logger.scalar_summary(f'train_{self.name}', 'lr', self.scheduler.get_lr()[0], epoch)

    @staticmethod
    def _get_init_message(metrics: List[str]):
        return '\t'.join(['Epoch', 'Train loss'] + [f'Valid {it}' for it in metrics])

    @staticmethod
    def _get_progress_message(epoch: int, train_loss: float, metrics: Dict[str, float]):
        message = f'{epoch:>5}\t{train_loss:>10.3f}\t'
        message += '\t'.join([f'{metrics[it]:.3f}'.rjust(len(it) + 6) for it in metrics])
        return message

    def _epoch_completed(self, engine: Engine):
        self._progress_bar.close()

        self._evaluator.run(self.data_loaders.valid)

        train_loss = self._train_loss.value()
        metrics = self._evaluator.state.metrics
        epoch = self._epoch + self.finished_epochs

        self.logger.info(self._get_progress_message(self._epoch, train_loss, metrics))
        self.logger.scalar_summary(f'train_{self.name}', 'loss', train_loss, epoch)

        for it in metrics:
            self.logger.scalar_summary(f'valid_{self.name}', it, metrics[it], epoch)

        if self.backup_path is not None:
            self.save_checkpoint(f'{self.backup_path}/checkpoint_{self.name}_{self._epoch:02}.pth')

    def save_checkpoint(self, path: str):
        state = {
            'epoch': self._epoch,
            'iteration': self._iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self._epoch = checkpoint['epoch']
        self._iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _register_handlers(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_STARTED, self._iteration_started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._iteration_completed)
        engine.add_event_handler(Events.EPOCH_STARTED, self._epoch_started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._epoch_completed)

    def train(self, num_epochs: int):
        self.logger.info(self._get_init_message(list(self.metrics.keys())))
        self._trainer.run(self.data_loaders.train, num_epochs)
