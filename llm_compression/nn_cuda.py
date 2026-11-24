"""Bounded-memory GPU input preparation and FP32 metric accumulation."""

import numpy as np


def device_batches(torch, messages, context, batch_size, *, chunk_bytes=1048576,
                   device='cuda', generator=None):
    """Upload compact byte chunks, then assemble/reset contexts on the device.

    Only a chunk of byte values/positions and one minibatch of contexts are
    resident. Message history survives chunk boundaries, never message boundaries.
    Chunk sizes are aligned to optimizer batches so changing chunk_bytes does
    not change the examples in an optimizer step. A generator shuffles examples
    within each batch on the device, without uploading expanded CPU contexts.
    """
    from .nn_training import byte_batches

    if context < 0 or batch_size < 1 or chunk_bytes < 1:
        raise ValueError('Context must be nonnegative; batch and chunk sizes must be positive')
    chunk_bytes = max(batch_size, chunk_bytes // batch_size * batch_size)
    history = torch.full((context,), 256, device=device, dtype=torch.int64)
    required = context - torch.arange(context, device=device)
    for data, positions in byte_batches(messages, chunk_bytes):
        # Copy read-only NumPy byte views before wrapping them with PyTorch.
        values = torch.from_numpy(np.array(data, copy=True)).to(device=device, dtype=torch.int64)
        positions = torch.from_numpy(positions).to(device=device)
        padded = torch.cat((history, values))
        for start in range(0, len(data), batch_size):
            count = min(batch_size, len(data) - start)
            x = padded[start:start + count + context].unfold(0, context + 1, 1)[:, :context]
            x = torch.where(positions[start:start + count, None] >= required, x, 256)
            y = values[start:start + count]
            if generator is not None:
                order = torch.randperm(count, generator=generator, device=device)
                x, y = x[order], y[order]
            yield x, y
        history = padded[-context:].clone() if context else padded[:0]


class Float32Sum:
    """Pairwise block reduction plus compensated summation, entirely in FP32.

    A running FP32 scalar loses small increments on long runs. Reducing fixed
    blocks first and compensating their sums avoids that drift without FP64 or
    a host synchronization per batch. Storage is bounded regardless of run size.
    """

    def __init__(self, torch, device, block_size=1024):
        if block_size < 1:
            raise ValueError('Accumulator block size must be positive')
        self.torch = torch
        self.buffer = torch.empty(block_size, device=device, dtype=torch.float32)
        self.total = torch.zeros((), device=device, dtype=torch.float32)
        self.correction = torch.zeros_like(self.total)
        self.count = 0

    def add(self, value):
        self.buffer[self.count].copy_(value.detach())
        self.count += 1
        if self.count == len(self.buffer):
            self._flush()

    def _flush(self):
        if self.count:
            increment = self.buffer[:self.count].sum() - self.correction
            updated = self.total + increment
            self.correction = (updated - self.total) - increment
            self.total = updated
            self.count = 0

    def item(self):
        self._flush()
        return self.total.item()
