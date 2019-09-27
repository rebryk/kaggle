from typing import Callable

import torch
from ignite.engine import Engine, _prepare_batch

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def create_supervised_trainer(model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              loss_fn: torch.nn.Module,
                              device: torch.cuda.device = None,
                              mixed_precision: bool = False,
                              non_blocking: bool = False,
                              prepare_batch: Callable = _prepare_batch) -> Engine:
    if device:
        model = model.to(device)

    def _process_function(engine: Engine, batch):
        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # TODO: register BACKWARD_PASS_COMPLETED event
        # TODO: fire BACKWARD_PASS_COMPLETED event

        optimizer.step()

        return loss.item()

    return Engine(_process_function)


def create_tester(model: torch.nn.Module,
                  device: torch.cuda.device = None,
                  non_blocking: bool = False) -> Engine:
    if device:
        model = model.to(device)

    def _process_function(engine: Engine, batch):
        model.eval()

        with torch.no_grad():
            x, labels_or_ids = batch
            x = x.to(device=device, non_blocking=non_blocking) if device else x
            y_pred = model(x)

        return labels_or_ids, y_pred

    return Engine(_process_function)
