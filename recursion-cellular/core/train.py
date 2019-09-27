from typing import Dict, List

import torch
from ignite.engine import create_supervised_evaluator, Engine, State, Events
from ignite.metrics import Metric
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.ignite import create_supervised_trainer
from utils import Logger, get_logger, create
from utils.model import get_norm, save_checkpoint


# TODO: add mixed precision
# TODO: add gradient clipping

class SupervisedTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loaders: Dict[str, DataLoader],
                 num_epochs: int,
                 metrics: Dict[str, Metric],
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 is_iteration_scheduler: bool = False,
                 device: torch.cuda.device = None,
                 checkpoint: str = None,
                 backup_path: str = None,
                 logger: Logger = None,
                 start_epoch: int = 0,
                 mixed_precision: bool = False,
                 stage: str = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loaders = loaders
        self.metrics = metrics

        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.end_epoch = start_epoch + num_epochs

        self.scheduler = scheduler
        self.is_iteration_scheduler = is_iteration_scheduler

        self.checkpoint = checkpoint
        self.backup_path = f'{backup_path}/{stage}'

        self.logger = logger or get_logger()
        self.stage = stage

        self._progress_bar = None
        self._description = 'TRAINING {}/{} - loss: {:.3f}'
        self._evaluator_description = 'VALIDATION {}/{}'

        self._trainer = create_supervised_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device=device,
            mixed_precision=mixed_precision,
            non_blocking=True
        )
        self._register_trainer_handlers(self._trainer)

        self._evaluator = create_supervised_evaluator(
            model=model,
            metrics=metrics,
            device=device,
            non_blocking=True
        )
        self._register_evaluator_handlers(self._evaluator)

    def run(self):
        metrics = list(self.metrics.keys())
        self.logger.info(self._get_init_message(metrics))
        self._trainer.run(self.loaders['train'], self.end_epoch)

    def _started(self, engine: Engine):
        epoch = self.start_epoch

        if self.checkpoint is not None:
            self.logger.info(f'Loading checkpoint: {self.checkpoint}')

            checkpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # Update start epoch
            epoch = checkpoint['epoch']

        engine.state = State(
            dataloader=self.loaders['train'],
            iteration=epoch * len(self.loaders['train']),
            epoch=epoch,
            max_epochs=self.end_epoch,
            metrics={}
        )

    def _iteration_started(self, engine: Engine):
        if self.scheduler is not None and self.is_iteration_scheduler:
            self.scheduler.step()
            self.logger.scalar_summary(
                logger_tag=self._get_tag('train'),
                tag='lr',
                value=self.scheduler.get_lr()[0],
                step=engine.state.iteration
            )

    def _iteration_completed(self, engine: Engine):
        train_loss = engine.state.output

        self.logger.scalar_summary(
            logger_tag=self._get_tag('train'),
            tag='iter_loss',
            value=train_loss,
            step=engine.state.iteration
        )

        self.logger.scalar_summary(
            logger_tag=self._get_tag('train'),
            tag='grad_norm',
            value=get_norm(self.model),
            step=engine.state.iteration
        )

        self._progress_bar.desc = self._description.format(self._trainer.state.epoch, self.end_epoch, train_loss)
        self._progress_bar.update(1)

    def _get_tag(self, tag: str) -> str:
        return f'{self.stage}/{tag}' if self.stage else tag

    def _epoch_started(self, engine: Engine):
        self._progress_bar = tqdm(
            initial=0,
            leave=False,
            total=len(self.loaders['train']),
            desc=self._description.format(self._trainer.state.epoch, self.end_epoch, 0)
        )

        if self.scheduler is not None and not self.is_iteration_scheduler:
            self.scheduler.step()
            self.logger.scalar_summary(
                logger_tag=self._get_tag('train'),
                tag='lr',
                value=self.scheduler.get_lr()[0],
                step=engine.state.iteration
            )

    def _evaluator_epoch_started(self, engine: Engine):
        self._progress_bar = tqdm(
            initial=0,
            leave=False,
            total=len(self.loaders['valid']),
            desc=self._evaluator_description.format(self._trainer.state.epoch, self.end_epoch)
        )

    def _evaluator_iteration_completed(self, engine: Engine):
        self._progress_bar.update(1)

    def _evaluator_epoch_completed(self, engine: Engine):
        self._progress_bar.close()

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

        self._evaluator.run(self.loaders['valid'])

        epoch = self._trainer.state.epoch
        train_loss = self._trainer.state.output
        metrics = self._evaluator.state.metrics

        self.logger.info(self._get_progress_message(epoch, train_loss, metrics))

        for metric in metrics:
            self.logger.scalar_summary(
                logger_tag=self._get_tag('valid'),
                tag=metric,
                value=metrics[metric],
                step=epoch
            )

        if self.backup_path is not None:
            create(self.backup_path)
            save_checkpoint(epoch=epoch,
                            model=self.model,
                            optimizer=self.optimizer,
                            path=f'{self.backup_path}/checkpoint_{epoch:02}.pth')

    def _register_trainer_handlers(self, engine: Engine):
        engine.add_event_handler(Events.STARTED, self._started)
        engine.add_event_handler(Events.ITERATION_STARTED, self._iteration_started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._iteration_completed)
        engine.add_event_handler(Events.EPOCH_STARTED, self._epoch_started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._epoch_completed)

    def _register_evaluator_handlers(self, engine: Engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self._evaluator_epoch_started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._evaluator_epoch_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._evaluator_iteration_completed)
