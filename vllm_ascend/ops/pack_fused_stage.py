# SPDX-License-Identifier: Apache-2.0
"""Pack two token-row tensors into one contiguous stage activation.

Used when a fused PP stage has to hand the next rank a single buffer made of
two row blocks (decode rows, then prefill rows, or any two already-ordered
halves). The kernel is a row-wise copy: block 0 is written first, block 1
follows. CPU and missing-Triton builds fall back to ``torch.cat``.
"""

from __future__ import annotations

import torch

try:
    from vllm.triton_utils import tl, triton
    from vllm.utils.torch_utils import direct_register_custom_op

    _HAS_PACK_KERNEL = True
except Exception:  # pragma: no cover - import varies by environment
    tl = None  # type: ignore
    triton = None  # type: ignore
    direct_register_custom_op = None  # type: ignore
    _HAS_PACK_KERNEL = False


if _HAS_PACK_KERNEL:

    @triton.jit
    def _pack_fused_stage_kernel(
        first_ptr,
        second_ptr,
        out_ptr,
        first_rows,
        total_rows,
        hidden,
        BLOCK: tl.constexpr,
    ):
        # One program walks many rows. A (rows, hidden/BLOCK) grid exceeds
        # the NPU launch limit (coreDim <= 65535) once a prefill is a few
        # thousand tokens wide.
        pid = tl.program_id(axis=0)
        num_programs = tl.num_programs(axis=0)
        for row in range(pid, total_rows, num_programs):
            col = tl.arange(0, BLOCK)
            for col_base in range(0, hidden, BLOCK):
                cols = col_base + col
                col_mask = cols < hidden
                src_is_second = row >= first_rows
                src_row = row - first_rows
                first_row = tl.where(src_is_second, 0, row)
                second_row = tl.where(src_is_second, src_row, 0)
                first_val = tl.load(
                    first_ptr + first_row * hidden + cols,
                    mask=col_mask & (row < first_rows),
                    other=0,
                )
                second_val = tl.load(
                    second_ptr + second_row * hidden + cols,
                    mask=col_mask & src_is_second,
                    other=0,
                )
                val = tl.where(src_is_second, second_val, first_val)
                tl.store(out_ptr + row * hidden + cols, val, mask=col_mask)

    def pack_fused_stage_rows(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim < 2 or second.ndim < 2:
            raise ValueError("pack_fused_stage_rows expects token-row tensors")
        if first.shape[1:] != second.shape[1:]:
            raise ValueError(
                f"row width mismatch: {tuple(first.shape)} vs {tuple(second.shape)}"
            )
        first = first.contiguous()
        second = second.contiguous()
        rows = int(first.shape[0]) + int(second.shape[0])
        out = torch.empty((rows, *first.shape[1:]), dtype=first.dtype, device=first.device)
        if rows == 0:
            return out
        hidden = int(first.numel() // first.shape[0])
        # View as [rows, hidden] so the kernel has one contiguous feature axis.
        flat_first = first.reshape(first.shape[0], hidden)
        flat_second = second.reshape(second.shape[0], hidden)
        flat_out = out.reshape(rows, hidden)
        block = 128
        # Stay far under the 65535 coreDim cap. Each program loops over rows.
        num_programs = min(rows, 128)
        _pack_fused_stage_kernel[(num_programs,)](
            flat_first,
            flat_second,
            flat_out,
            int(first.shape[0]),
            rows,
            hidden,
            BLOCK=block,
        )
        return out

    def pack_fused_stage_rows_fake(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        rows = int(first.shape[0]) + int(second.shape[0])
        return torch.empty((rows, *first.shape[1:]), dtype=first.dtype, device=first.device)

    direct_register_custom_op(
        op_name="pack_fused_stage_rows",
        op_func=pack_fused_stage_rows,
        fake_impl=pack_fused_stage_rows_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )


def pack_token_rows(parts: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate token-row blocks, using the custom op on NPU."""
    if not parts:
        raise ValueError("pack_token_rows received no tensors")
    out = parts[0]
    for part in parts[1:]:
        if out.numel() == 0:
            out = part
            continue
        if part.numel() == 0:
            continue
        use_op = (
            _HAS_PACK_KERNEL
            and out.device.type != "cpu"
            and hasattr(torch.ops, "vllm")
            and hasattr(torch.ops.vllm, "pack_fused_stage_rows")
        )
        if use_op:
            out = torch.ops.vllm.pack_fused_stage_rows(out, part)
        else:
            out = torch.cat((out, part), dim=0)
    return out
