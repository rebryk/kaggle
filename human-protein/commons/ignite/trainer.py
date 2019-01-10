from typing import Callable

import torch
from apex.fp16_utils import FP16_Optimizer
from ignite.engine import Engine, _prepare_batch
from torch.nn.utils import clip_grad_norm_


def create_supervised_trainer(model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              loss_fn: torch.nn.Module,
                              max_norm: float = None,
                              norm_type: int = 2,
                              device: torch.cuda.device = None,
                              non_blocking: bool = False,
                              mixed_precision: bool = False,
                              static_loss_scale: int = 512,
                              prepare_batch: Callable = _prepare_batch) -> Engine:
    if device:
        model.to(device)

    if mixed_precision:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=static_loss_scale)

    def _process_function(engine: Engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if mixed_precision:
            optimizer.backward(loss)
        else:
            loss.backward()

        if max_norm:
            clip_grad_norm_(model.parameters(), max_norm, norm_type)

        optimizer.step()
        return loss.item()

    return Engine(_process_function)
