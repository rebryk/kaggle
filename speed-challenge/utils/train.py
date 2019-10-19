import os
import sys
from typing import Dict
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from apex import amp
from ignite.engine import Engine, State, Events
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.logger import get_logger
from utils.model import save_checkpoint, load_checkpoint, get_grad_norm
from utils.path import create
from utils.time import Timer
from utils.utils import get_memory_usage


class SupervisedTrainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 loaders: Dict[str, DataLoader],
                 num_epochs: int,
                 n_gpu: int,
                 is_master: bool,
                 local_rank: int,
                 backup_path: str,
                 scheduler: _LRScheduler = None,
                 is_iteration_scheduler: bool = False,
                 max_grad_norm: float = None,
                 mixed_precision: bool = False,
                 gradient_accumulation_steps: int = 1,
                 checkpoint: str = None,
                 log_interval: int = 1,
                 checkpoint_interval: int = 1,
                 start_epoch: int = 0,
                 stage: str = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loaders = loaders
        self.num_epochs = num_epochs
        self.n_gpu = n_gpu
        self.is_master = is_master
        self.local_rank = local_rank
        self.multi_gpu = n_gpu > 1
        self.scheduler = scheduler
        self.is_iteration_scheduler = is_iteration_scheduler
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint = checkpoint
        self.backup_path = f'{backup_path}/{stage}' if stage else backup_path
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self._max_iteration = len(loaders['train'])
        self._last_group_size = self._max_iteration % gradient_accumulation_steps

        # Store the model gradients norm to log it into TensorBoard
        self._last_grad_norm = 0.0

        # Configure training epochs
        self.start_epoch = start_epoch
        self.end_epoch = start_epoch + num_epochs

        # Initialize train engine
        self._trainer = Engine(self._process_batch_train)
        self._register_trainer_handlers(self._trainer)

        # Initialize evaluation engine
        self._evaluator = Engine(self._process_batch_eval)
        self._register_evaluator_handlers(self._evaluator)

        # Progress bar descriptions
        self._progress_bar = None
        self._trainer_description = 'TRAINING {}/{} - loss: {:.3f}'
        self._evaluator_description = 'VALIDATION {}/{} - loss: {:.3f}'

        # Initialize logger and tensorboardX
        self._timer = Timer()
        self._logger = get_logger(self.__class__.__name__, stream=sys.stdout if local_rank == 0 else None)

        if self.is_master:
            self._trainer_tensorboard = SummaryWriter(log_dir=os.path.join(self.backup_path, 'train'))
            self._evaluator_tensorboard = SummaryWriter(log_dir=os.path.join(self.backup_path, 'valid'))

    def train(self):
        self._trainer.run(self.loaders['train'], self.end_epoch)

    def _process_batch_train(self, engine: Engine, batch: Tuple):
        """Processes a single batch in training mode."""

        with self._timer:
            is_step_iteration = engine.state.local_iteration % self.gradient_accumulation_steps == 0
            is_last_batch = engine.state.local_iteration == self._max_iteration
            is_last_group = engine.state.local_iteration >= self._max_iteration - self._last_group_size + 1

            if self.n_gpu > 0:
                batch = tuple(it.to(f'cuda:{self.local_rank}') for it in batch)

            x, y = batch
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            if self.gradient_accumulation_steps > 1:
                # If the last group size is less than gradient_accumulation_steps we should use another denominator
                if is_last_group:
                    loss = loss / self._last_group_size
                else:
                    loss = loss / self.gradient_accumulation_steps

            if self.mixed_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Average loss across all GPUs
            if self.multi_gpu:
                dist.all_reduce(loss)
                loss /= self.n_gpu

            if is_step_iteration or is_last_batch:
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self._last_grad_norm = get_grad_norm(self.model)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Do reverse transformation
            return self.gradient_accumulation_steps * loss.item()

    def _process_batch_eval(self, engine: Engine, batch: Tuple):
        """Processes a single batch in evaluation mode."""

        with self._timer:
            with torch.no_grad():
                if self.n_gpu > 0:
                    batch = tuple(it.to(f'cuda:{self.local_rank}') for it in batch)

                x, y = batch
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

                # Average loss across all GPUs
                if self.multi_gpu:
                    dist.all_reduce(loss)
                    loss /= self.n_gpu

                return loss.item()

    def _trainer_started(self, engine: Engine):
        epoch = self.start_epoch

        if self.checkpoint is not None:
            self._logger.info(f'Loading checkpoint: {self.checkpoint}')
            epoch = load_checkpoint(self.checkpoint, self.model, self.optimizer, self.scheduler)

        if self.n_gpu > 0:
            self._logger.info(f'Moving the model to GPU {self.local_rank}')
            self.model = self.model.to(f'cuda:{self.local_rank}')

        if self.mixed_precision:
            self._logger.info('Use mixed precision training')
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        if self.multi_gpu:
            self._logger.info('Using nn.parallel.DistributedDataParallel for distributed training')
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )

        # Print the progress bar initial message
        self._logger.info(self._get_init_message())

        # Update the engine state
        engine.state = State(
            dataloader=self.loaders['train'],
            iteration=epoch * len(self.loaders['train']),
            epoch=epoch,
            max_epochs=self.end_epoch,
            metrics={}
        )

    def _trainer_epoch_started(self, engine: Engine):
        self.model.train()
        self._timer.reset()

        # Initialize new variable
        engine.state.average_loss = 0
        engine.state.local_iteration = 0

        if self.multi_gpu:
            torch.distributed.barrier()

        self._progress_bar = tqdm(
            initial=0,
            leave=False,
            total=len(self.loaders['train']),
            desc=self._trainer_description.format(self._trainer.state.epoch, self.end_epoch, 0)
        )

        if self.scheduler is not None and not self.is_iteration_scheduler:
            self.scheduler.step()

    def _trainer_iteration_started(self, engine: Engine):
        engine.state.local_iteration = engine.state.local_iteration + 1

        if self.scheduler is not None and self.is_iteration_scheduler:
            self.scheduler.step()

    def _trainer_iteration_completed(self, engine: Engine):
        # Update epoch average loss
        train_loss = engine.state.output
        iteration = engine.state.local_iteration
        engine.state.average_loss = engine.state.average_loss + (train_loss - engine.state.average_loss) / iteration

        # Update progress bar message
        self._progress_bar.desc = self._trainer_description.format(
            self._trainer.state.epoch,
            self.end_epoch,
            train_loss
        )
        self._progress_bar.update(1)

        if self.is_master and self._trainer.state.iteration % self.log_interval == 0:
            self._log_trainer_tensorboard()
            self._timer.reset()

    def _trainer_epoch_completed(self, engine: Engine):
        epoch = self._trainer.state.epoch
        self._progress_bar.close()

        if self.local_rank == 0 and epoch % self.checkpoint_interval == 0:
            create(f'{self.backup_path}/checkpoints')
            save_checkpoint(
                path=f'{self.backup_path}/checkpoints/checkpoint_{epoch:02}.pth',
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )

        # Run evaluation
        self._evaluator.run(self.loaders['valid'])

        train_loss = self._trainer.state.average_loss
        valid_loss = self._evaluator.state.average_loss
        self._logger.info(self._get_progress_message(epoch, train_loss, valid_loss))

        if self.is_master:
            self._trainer_tensorboard.add_scalar(tag='losses/loss', scalar_value=train_loss, global_step=epoch)
            self._evaluator_tensorboard.add_scalar(tag='losses/loss', scalar_value=valid_loss, global_step=epoch)

    def _evaluator_epoch_started(self, engine: Engine):
        self.model.eval()
        self._timer.reset()

        # Initialize new variable
        engine.state.average_loss = 0

        self._progress_bar = tqdm(
            initial=0,
            leave=False,
            total=len(self.loaders['valid']),
            desc=self._evaluator_description.format(self._trainer.state.epoch, self.end_epoch, 0.0)
        )

    def _evaluator_iteration_completed(self, engine: Engine):
        # Calculate epoch average loss
        valid_loss = engine.state.output
        iteration = engine.state.iteration
        engine.state.average_loss = engine.state.average_loss + (valid_loss - engine.state.average_loss) / iteration

        self._progress_bar.desc = self._evaluator_description.format(
            self._trainer.state.epoch,
            self.end_epoch,
            valid_loss
        )
        self._progress_bar.update(1)

        if self.is_master and iteration % self.log_interval == 0:
            self._log_evaluator_tensorboard()
            self._timer.reset()

    def _evaluator_epoch_completed(self, engine: Engine):
        self._progress_bar.close()

    def _register_trainer_handlers(self, engine: Engine):
        """Adds event handlers to train ignite engine."""

        engine.add_event_handler(Events.STARTED, self._trainer_started)
        engine.add_event_handler(Events.ITERATION_STARTED, self._trainer_iteration_started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._trainer_iteration_completed)
        engine.add_event_handler(Events.EPOCH_STARTED, self._trainer_epoch_started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._trainer_epoch_completed)

    def _register_evaluator_handlers(self, engine: Engine):
        """Adds event handlers to valid ignite engine."""

        engine.add_event_handler(Events.EPOCH_STARTED, self._evaluator_epoch_started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._evaluator_epoch_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._evaluator_iteration_completed)

    def _log_trainer_tensorboard(self):
        """Logs different statistics to TensorBoard."""

        iteration = self._trainer.state.iteration

        self._trainer_tensorboard.add_scalar('losses/iter_loss', self._trainer.state.output, iteration)
        self._trainer_tensorboard.add_scalar('global/speed', self._timer.average, iteration)
        self._trainer_tensorboard.add_scalar('global/memory', get_memory_usage(), iteration)
        self._trainer_tensorboard.add_scalar('global/grad_norm', self._last_grad_norm, iteration)

        if self.scheduler is not None:
            self._trainer_tensorboard.add_scalar('global/lr', self.scheduler.get_lr()[0], iteration)

    def _log_evaluator_tensorboard(self):
        """Logs different statistics to TensorBoard."""

        iteration = self._evaluator.state.iteration

        self._evaluator_tensorboard.add_scalar('global/speed', self._timer.average, iteration)
        self._evaluator_tensorboard.add_scalar('global/memory', get_memory_usage(), iteration)

    @staticmethod
    def _get_init_message():
        return '\t'.join(['Epoch', 'Train loss', 'Valid loss'])

    @staticmethod
    def _get_progress_message(epoch: int, train_loss: float, valid_loss: float):
        return f'{epoch:>5}\t{train_loss:>10.3f}\t{valid_loss:>10.3f}'
