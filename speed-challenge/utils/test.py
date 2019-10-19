import sys
from typing import Tuple, Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_logger, load_checkpoint


def _default_fold(result: np.array, output: Tuple) -> Tuple:
    ids, y_pred = output
    ids = ids.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return np.concatenate([result[0], ids]), np.concatenate([result[1], y_pred])


class Tester:
    def __init__(self,
                 model: torch.nn.Module,
                 checkpoint: str,
                 n_gpu: int,
                 is_master: bool,
                 local_rank: int,
                 init: Any = ([], []),
                 fold: Callable = _default_fold):
        self.model = model
        self.checkpoint = checkpoint
        self.n_gpu = n_gpu
        self.is_master = is_master
        self.local_rank = local_rank
        self.init = init
        self.fold = fold

        self.loader = None
        self.output = None

        # Initialize tester engine
        self._tester = Engine(self._process_batch)
        self._register_tester_handlers(self._tester)

        # Progress bar descriptions
        self._progress_bar = None
        self._description = 'TESTING'

        # Initialize logger
        self._logger = get_logger(self.__class__.__name__, stream=sys.stdout if local_rank == 0 else None)

        if self.checkpoint is not None:
            epoch = load_checkpoint(self.checkpoint, self.model)
            self._logger.info(f'Uploaded model from epoch {epoch}')

        if self.n_gpu > 0:
            self._logger.info(f'Moving the model to GPU {self.local_rank}')
            self.model = self.model.to(f'cuda:{self.local_rank}')

    def __call__(self, loader: DataLoader) -> Any:
        self.loader = loader
        self.output = self.init
        self._tester.run(loader, 1)
        return self.output

    def _process_batch(self, engine: Engine, batch: Tuple):
        """Processes a single batch in evaluation mode."""

        with torch.no_grad():
            if self.n_gpu > 0:
                batch = tuple(it.to(f'cuda:{self.local_rank}') for it in batch)

            x, labels_or_ids = batch
            y_pred = self.model(x)

            return labels_or_ids, y_pred

    def _epoch_started(self, engine: Engine):
        self.model.eval()

        self._progress_bar = tqdm(
            initial=0,
            leave=False,
            total=len(self.loader),
            desc=self._description
        )

    def _epoch_completed(self, engine: Engine):
        self._progress_bar.close()
        self._logger.info('Inference is finished')

    def _iteration_completed(self, engine: Engine):
        # TODO: use gather instead of all_gather

        if self.n_gpu > 1:
            labels_or_ids, y_pred = engine.state.output

            # Gather labels_or_ids from all GPUs
            gather_labels_or_ids = [torch.ones_like(labels_or_ids) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_labels_or_ids, labels_or_ids)
            labels_or_ids = torch.cat(gather_labels_or_ids, dim=0)

            # Gather y_preds from all GPUs
            gather_y_pred = [torch.ones_like(y_pred) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_y_pred, y_pred)
            y_pred = torch.cat(gather_y_pred, dim=0)

            engine.state.output = labels_or_ids, y_pred

        if self.is_master:
            self.output = self.fold(self.output, engine.state.output)

        if self.n_gpu > 1:
            torch.distributed.barrier()

        self._progress_bar.update(1)

    def _register_tester_handlers(self, engine: Engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self._epoch_started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._epoch_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._iteration_completed)
