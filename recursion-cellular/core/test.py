from typing import Callable, Any

import torch
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.ignite import create_tester
from utils import Logger, get_logger


class Tester:
    def __init__(self,
                 model: torch.nn.Module,
                 init: Any = None,
                 fold: Callable = None,
                 device: torch.cuda.device = None,
                 logger: Logger = None):
        self.model = model
        self.loader = None
        self.init = init
        self.output = None
        self.fold = fold
        self.logger = logger or get_logger()

        self._progress_bar = None
        self._description = 'TESTING'

        self._tester = create_tester(model=model, device=device)
        self._register_tester_handlers(self._tester)

    def __call__(self, loader: DataLoader) -> Any:
        self.loader = loader
        self.output = self.init
        self._tester.run(loader, 1)
        return self.output

    def _epoch_started(self, engine: Engine):
        self._progress_bar = tqdm(
            initial=0,
            leave=False,
            total=len(self.loader),
            desc=self._description
        )

    def _epoch_completed(self, engine: Engine):
        self._progress_bar.close()
        self.logger.info('Inference is finished')

    def _iteration_completed(self, engine: Engine):
        self.output = self.fold(self.output, engine.state.output)
        self._progress_bar.update(1)

    def _register_tester_handlers(self, engine: Engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self._epoch_started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._epoch_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._iteration_completed)
