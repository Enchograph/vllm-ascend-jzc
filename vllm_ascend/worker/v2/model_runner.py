# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import contextmanager
from dataclasses import replace
import os

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import (
    ExecuteModelState,
    GPUModelRunner,
    sort_batch_req_ids,
)

from vllm.v1.core.layered_prefill import (
    LayeredPrefillStateStore,
    LayeredForwardOutput,
    make_pp_aligned_layer_group_ranges,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import get_uniform_token_count
from vllm.model_executor.models.utils import extract_layer_index
from vllm.forward_context import BatchDescriptor, get_forward_context

from vllm_ascend.worker.v2.layered_prefill import (
    LayeredPPHandlerCapture,
    LayeredV2ExecuteModelState,
    concat_req_frontiers,
    detach_execute_model_state,
    empty_layered_prefill_counters,
    pad_activation_rows,
    slice_req_activations,
    split_d_p_req_ids,
    store_req_frontiers,
    subset_scheduler_output,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mc2_tokens_capacity,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
    set_ascend_forward_context,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.utils import (
    enable_dsa_cp,
    enable_sp,
    set_potential_max_tokens,
    vllm_version_is,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

if not vllm_version_is("0.27.1"):
    from vllm.v1.worker.gpu.model_runner import BatchReqState

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.eplb import AscendEPLBController
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    execute_model_state: ExecuteModelState | None

    @property
    def pcp_manager_cls(self) -> type[AscendPCPManager]:
        return AscendPCPManager

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        # FusedMoE can be constructed by the parent initializer and reads this
        # capacity while setting up MC2 communication.
        set_potential_max_tokens(vllm_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("Decode Context parallelism is not supported by Ascend NPU model runner v2.")

        with torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        self.use_aclgraph = (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and self.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not self.model_config.enforce_eager
        )
        load_collection_phase = self.ascend_config.eplb_config.load_collection_phase
        self.eplb = AscendEPLBController(
            parallel_config,
            device,
            load_collection_phase=(load_collection_phase if parallel_config.enable_eplb else "all"),
        )

        self.update_stream = None
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream = torch.npu.Stream()

        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.req_states
        del self.input_buffers
        del self.speculator

        # we define AscendEagleSpeculator in vllm_ascend.worker.v2.spec_decode.eagle.speculator
        # init_speculator will return AscendEagleSpeculator when eagle is used.
        # so here we just call init_speculator to reinitialize speculator.
        self.speculator: AscendEagleSpeculator | None = None
        if self.speculative_config is not None:
            self.speculator = init_speculator(self.vllm_config, self.device)
            # Shared update_stream: main model (ModelAclGraphManager) and draft
            # (Eagle/DFlash/DSpark AclGraphManager) all use this same stream.
            self.speculator.update_stream = self.update_stream

        # AscendRequestState has extra `num_computed_tokens_cpu` attribute.
        # so reinitialize req_states here.
        self.req_states: AscendRequestState = AscendRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        # Layered Prefill V2: second buffer set for the P sub-batch.  V2's
        # prepare_inputs writes through self.input_buffers views, so D and P
        # must not share storage.  See layered_prefill_v2_migration_plan.md §3.1.
        layered_cfg = self.ascend_config.scheduler_config.layered_prefill_config
        self._layered_prefill_enabled = bool(layered_cfg.enabled)
        self._fuse_mixed_batch = bool(getattr(layered_cfg, "fuse_mixed_batch", False))
        self._same_layer_batch = bool(getattr(layered_cfg, "same_layer_batch", False))
        layered_ext = getattr(
            self.ascend_config.scheduler_config, "layered_prefill_extensions", None
        )
        self._skip_relay_stages = bool(getattr(layered_ext, "skip_relay_stages", False))
        # True after load_model wires the D→P adapter (PP=1 and PP>1).
        self._layered_prefill_v2_ready = False
        self.layered_prefill_state = LayeredPrefillStateStore()
        self.layered_prefill_model_adapter = None
        self._layered_skip_pp_decode_update = False
        # Set when a step is pure relay, so sample_tokens can answer without
        # an ExecuteModelState.
        self._layered_relay_output: ModelRunnerOutput | None = None
        self._layered_input_buffers: AscendInputBuffers | None = None
        self.layered_prefill_counters = empty_layered_prefill_counters()
        if self._layered_prefill_enabled:
            # Phase 1 schedules a single P request; size by max query tokens.
            # Keep max_num_reqs aligned with the main runner so query_start_loc
            # padding helpers stay valid if the scheduler grows past k=1.
            self._layered_input_buffers = AscendInputBuffers(
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                device=self.device,
            )
        logger.info(
            "NPUModelRunner layered_enabled=%s additional_config_is_dict=%s",
            self._layered_prefill_enabled,
            isinstance(getattr(vllm_config, "additional_config", None), dict),
        )

        # we need to copy num_computed_tokens back to cpu to help
        # update actual seq_lens_cpu. gpu attention backend doesn't need these
        # attributes, cause their attention backends doesn't use seq_lens_cpu.
        # and seq_lens_cpu is deprecated in gpu_model_runner_v2.
        self.num_computed_tokens_event = torch.npu.Event()
        self.num_computed_tokens_stream = torch.npu.Stream()
        self.num_computed_tokens_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        # NOTE: In GPUModelRunner, decode_query_len is initialized in load_model(),
        # +1 is hardcoded here but not in vllm.
        self.decode_query_len = self.num_speculative_steps + 1
        # Set _mc2_tokens_capacity and _reserved_mc2_mask for MoE communication optimization.
        # TODO: remove set_cos_and_sin (together with update_cos_sin) when mla can properly handle cos/sin internally
        set_cos_and_sin(vllm_config, self.max_num_reqs, self.decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        set_potential_max_tokens(vllm_config)

    @contextmanager
    def _layered_p_buffers(self):
        """Swap in the P-only AscendInputBuffers for the Prefill sub-batch.

        Must not fork prepare_inputs: that path hard-codes self.input_buffers.
        D always uses the main buffers; P runs under this context (eager only).
        """
        if self._layered_input_buffers is None:
            raise RuntimeError(
                "Layered Prefill P buffers were not allocated; "
                "enable layered_prefill_config before constructing NPUModelRunner"
            )
        main = self.input_buffers
        self.input_buffers = self._layered_input_buffers
        try:
            yield
        finally:
            self.input_buffers = main

    @contextmanager
    def _layered_pp_handler_capture(self):
        """Defer PPHandler so D and final-P share one sampled-token slot."""
        handler = self.pp_handler
        if handler is None:
            yield None
            return
        capture = LayeredPPHandlerCapture(handler)
        self.pp_handler = capture
        try:
            yield capture
        finally:
            self.pp_handler = handler

    def update_pp_decode_requests(self):
        # Layered D still calls super().execute_model, which would consume the
        # one-slot queue a second time.  Lifecycle already applied it first.
        if self._layered_skip_pp_decode_update:
            return
        super().update_pp_decode_requests()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        with graph_manager_wrapper(self):
            super().initialize_kv_cache(kv_cache_config)
            if self.pcp_manager is not None:
                assert isinstance(self.pcp_manager, AscendPCPManager)
                self.pcp_manager.vllm_config = self.vllm_config

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ):
        layered_plan = getattr(scheduler_output, "layered_prefill_plan", None)
        if layered_plan is not None and not dummy_run:
            if not self._layered_prefill_v2_ready:
                raise NotImplementedError(
                    "layered_prefill_config on Model Runner V2 is not ready: "
                    "load_model did not enable the D→P path"
                )
            return self._execute_layered_step(
                scheduler_output,
                layered_plan,
                intermediate_tensors=intermediate_tensors,
            )
        if self._layered_prefill_enabled and not dummy_run:
            logger.info_once(
                "Layered Prefill V2 execute_model saw no plan (regular path)"
            )
        if vllm_version_is("0.27.1"):
            return super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
            )
        return super().execute_model(
            scheduler_output,
            intermediate_tensors=intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
            context_len=context_len,
        )

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        if self._layered_prefill_enabled:
            from vllm_ascend.models.layered_prefill import (
                create_layered_prefill_model_adapter,
            )

            try:
                self.layered_prefill_model_adapter = create_layered_prefill_model_adapter(
                    self.model
                )
            except TypeError as error:
                raise RuntimeError(
                    "The loaded model does not support layered prefill"
                ) from error
            # Orchestration (V2 milestone) is wired; flip the fail-closed gate.
            self._layered_prefill_v2_ready = True
            logger.info(
                "Layered Prefill V2 adapter ready: %s",
                type(self.layered_prefill_model_adapter).__name__,
            )

    def _execute_layered_step(
        self,
        scheduler_output: SchedulerOutput,
        layered_plan,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        """D then P dual sub-batch orchestration (V2, including PP>1).

        Default execution order D→P is a correctness constraint: Ascend stores
        model_state.attn_metadata as instance state consumed by
        ModelAclGraphManager.run_fullgraph (see migration plan §3.1).

        ``fuse_mixed_batch``: skip the D graph and run one eager layer-group
        forward on the combined D+P batch. Decode tokens sample only on
        ``is_sampling_step``. Pure decode steps still use the graph. PP>1
        sends that mixed activation as a single IntermediateTensors payload
        (no D/P split).

        ``same_layer_batch`` (PP=1 only): D finishes each round. D runs
        layers ``[0, group_start)`` alone, batches with P on the active
        group, then finishes ``[group_end, L)`` and samples. P pauses after
        the group and keeps its frontier. Pure D still uses the FULL graph.

        PP>1 reuses the V1 outer-worker protocol: one recv/send per scheduler
        step, with D/P rows packed into a single IntermediateTensors payload
        plus layered_pp_{d,p}_rows metadata.  Sampled tokens share one
        PPHandler slot (D ∪ final P); intermediate P groups are excluded.
        """
        if self.layered_prefill_model_adapter is None:
            raise RuntimeError(
                "The loaded model does not have a layered prefill adapter"
            )

        use_pp = self.parallel_config.pipeline_parallel_size > 1
        if not use_pp and intermediate_tensors is not None:
            raise RuntimeError(
                "Layered Prefill V2 received PP intermediate tensors at PP=1"
            )

        finished = getattr(scheduler_output, "finished_req_ids", ()) or ()
        preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
        self.layered_prefill_state.clear_many(finished)
        if preempted:
            self.layered_prefill_state.clear_many(preempted)

        d_req_ids, p_req_ids = split_d_p_req_ids(scheduler_output, layered_plan)
        fuse_mixed = (
            bool(getattr(self, "_fuse_mixed_batch", False))
            and bool(d_req_ids)
            and bool(p_req_ids)
        )
        same_layer = (
            bool(getattr(self, "_same_layer_batch", False))
            and not use_pp
            and bool(d_req_ids)
            and bool(p_req_ids)
        )
        if same_layer and fuse_mixed:
            raise RuntimeError(
                "same_layer_batch and fuse_mixed_batch cannot run together"
            )
        # A stage with no work in this layer group relays it instead of
        # running the runner machinery.  Decided before the activation is
        # unpacked, so a stage ahead of the owner does not wait for a payload
        # it will never read.  The D/P-split path is excluded: it carries two
        # payloads, and relaying one of them needs its own reasoning.
        relay_action = None
        if use_pp and (fuse_mixed or not d_req_ids):
            relay_action = self._layered_relay_stage_action(layered_plan)

        mixed_intermediate = None
        d_intermediate = None
        p_intermediate = None
        if use_pp and not self.is_first_pp_rank and relay_action is None:
            if fuse_mixed:
                mixed_intermediate = intermediate_tensors
                if mixed_intermediate is None:
                    raise RuntimeError(
                        "Fused mixed PP stage did not receive activation"
                    )
            else:
                d_intermediate, p_intermediate = self._split_layered_pp_intermediate(
                    intermediate_tensors
                )
        self._record_layered_step(
            layered_plan,
            n_d=len(d_req_ids),
            n_p=len(p_req_ids),
            fused_mixed=fuse_mixed,
            same_layer=same_layer,
        )
        logger.info(
            "Layered Prefill V2 step group=%s/%s layers=[%s,%s) "
            "n_d=%s n_p=%s final_group=%s sampling=%s pp=%s "
            "fused_mixed=%s same_layer=%s",
            layered_plan.group_id,
            layered_plan.num_groups,
            layered_plan.group_start,
            layered_plan.group_end,
            len(d_req_ids),
            len(p_req_ids),
            bool(layered_plan.is_final_group),
            bool(getattr(layered_plan, "is_sampling_step", layered_plan.is_final_group)),
            int(self.parallel_config.pipeline_parallel_size),
            fuse_mixed,
            same_layer,
        )
        if (
            use_pp
            and not self.is_first_pp_rank
            and d_req_ids
            and d_intermediate is None
            and not fuse_mixed
        ):
            raise RuntimeError("Layered PP D sub-batch did not receive activation")

        # One scheduler step → one lifecycle pass.  Must cover *all* new/cached
        # rows (including the P request) before either sub-batch forward.
        # Consume T-pp_size sampled tokens *before* finish/add so the generation
        # counters still match the PendingRecv snapshot; D's super().execute_model
        # then skips the same call via _layered_skip_pp_decode_update.
        self._layered_skip_pp_decode_update = False
        self._apply_scheduler_lifecycle(scheduler_output)
        self._layered_skip_pp_decode_update = True
        try:
            # Ascend builds seq_lens from num_computed_tokens_cpu.  That mirror
            # is normally refreshed in prepare_inputs only for
            # scheduled_cached_reqs rows; D/P sub-batches strip those fields
            # to avoid double update_requests, so sync explicitly here.
            self._sync_ascend_num_computed_tokens_cpu(
                list(scheduler_output.num_scheduled_tokens)
            )

            if same_layer:
                return self._execute_same_layer_d_complete(
                    scheduler_output,
                    layered_plan,
                    d_req_ids,
                    p_req_ids,
                )

            if relay_action is not None:
                return self._execute_layered_relay_step(
                    scheduler_output,
                    relay_action,
                    intermediate_tensors,
                    fused=fuse_mixed,
                )

            if fuse_mixed:
                mixed_output = self._strip_lifecycle_fields(scheduler_output)
                with self._layered_p_buffers():
                    p_state, p_layered_output = self._run_layered_prefill_subbatch(
                        mixed_output, layered_plan, mixed_intermediate
                    )
                torch.npu.synchronize()
                self.execute_model_state = LayeredV2ExecuteModelState(
                    scheduler_output=scheduler_output,
                    d_state=None,
                    p_state=p_state,
                    sample_p=bool(
                        getattr(
                            layered_plan,
                            "is_sampling_step",
                            layered_plan.is_final_group,
                        )
                    ),
                    fused_mixed=True,
                )
                if use_pp and not self.is_last_pp_rank:
                    packed = self._layered_fused_pp_payload(
                        layered_plan, p_layered_output
                    )
                    packed.kv_connector_output = getattr(
                        p_layered_output, "kv_connector_output", None
                    )
                    return packed
                return None

            d_state: ExecuteModelState | None = None
            pp_intermediates: list[IntermediateTensors] = []
            # D before P: Ascend stores attn_metadata on model_state for fullgraph.
            if d_req_ids:
                d_output = subset_scheduler_output(
                    scheduler_output,
                    d_req_ids,
                    layered_plan=None,
                    include_one_time_updates=False,
                )
                # Lifecycle already applied; strip rows that would re-enter it.
                d_output = self._strip_lifecycle_fields(d_output)
                result = super().execute_model(
                    d_output, intermediate_tensors=d_intermediate
                )
                d_state = self.execute_model_state
                self.execute_model_state = None
                if d_state is None:
                    raise RuntimeError(
                        "Layered Prefill V2 D sub-batch missing execute state"
                    )
                d_cg = CUDAGraphMode.NONE
                try:
                    runtime_mode = get_forward_context().cudagraph_runtime_mode
                    if runtime_mode is not None:
                        d_cg = runtime_mode
                except Exception:
                    pass
                logger.info_once(
                    "Layered Decode subbatch selected cudagraph_mode=%s",
                    d_cg.name,
                )
                if use_pp and not self.is_last_pp_rank:
                    if not isinstance(result, IntermediateTensors):
                        raise RuntimeError(
                            "Layered Prefill V2 D sub-batch must return "
                            "IntermediateTensors on a non-last PP rank"
                        )
                    if p_req_ids:
                        torch.npu.synchronize()
                        result = self._clone_intermediate_tensors(result)
                    pp_intermediates.append(result)
                elif result is not None:
                    raise RuntimeError(
                        "Layered Prefill V2 D sub-batch returned an unexpected output"
                    )
                if p_req_ids:
                    # P reuses model activation scratch; keep D logits inputs alive.
                    d_state = detach_execute_model_state(d_state)
                    # Establish a stream boundary before P may pick a different
                    # MoE backend / attention workspace.
                    torch.npu.synchronize()

            p_state: ExecuteModelState | None = None
            p_layered_output: LayeredForwardOutput | None = None
            if p_req_ids:
                p_output = subset_scheduler_output(
                    scheduler_output,
                    p_req_ids,
                    layered_plan=layered_plan,
                    include_one_time_updates=False,
                )
                with self._layered_p_buffers():
                    p_state, p_layered_output = self._run_layered_prefill_subbatch(
                        p_output, layered_plan, p_intermediate
                    )
                torch.npu.synchronize()
                if use_pp and not self.is_last_pp_rank:
                    assert p_layered_output is not None
                    pp_intermediates.append(
                        self.layered_prefill_model_adapter.to_intermediate_tensors(
                            p_layered_output
                        )
                    )

            self.execute_model_state = LayeredV2ExecuteModelState(
                scheduler_output=scheduler_output,
                d_state=d_state,
                p_state=p_state,
                sample_p=bool(
                    getattr(layered_plan, "is_sampling_step", layered_plan.is_final_group)
                ),
            )
            if use_pp and not self.is_last_pp_rank:
                d_pp = pp_intermediates[0] if d_req_ids else None
                p_pp = pp_intermediates[-1] if p_req_ids else None
                if d_pp is None and p_pp is None:
                    raise RuntimeError(
                        "Layered Prefill V2 non-last PP rank produced no intermediates"
                    )
                packed = self._combine_layered_pp_intermediate(d_pp, p_pp)
                packed.kv_connector_output = (
                    getattr(d_pp, "kv_connector_output", None)
                    if d_pp is not None
                    else None
                )
                if packed.kv_connector_output is None and p_pp is not None:
                    packed.kv_connector_output = getattr(
                        p_pp, "kv_connector_output", None
                    )
                return packed
            return None
        finally:
            self._layered_skip_pp_decode_update = False

    def _num_hidden_layers(self) -> int:
        cfg = self.model_config
        hf = getattr(cfg, "hf_text_config", None) or getattr(cfg, "hf_config", None)
        value = getattr(hf, "num_hidden_layers", None)
        if value is None:
            value = getattr(cfg, "num_hidden_layers", None)
        if value is None:
            raise RuntimeError("same_layer_batch needs model num_hidden_layers")
        return int(value)

    def _execute_same_layer_d_complete(
        self,
        scheduler_output: SchedulerOutput,
        layered_plan,
        d_req_ids: list[str],
        p_req_ids: list[str],
    ):
        """D-head / same-layer mix / D-tail. Sample D every step. PP=1 only."""
        num_layers = self._num_hidden_layers()
        group_start = int(layered_plan.group_start)
        group_end = int(layered_plan.group_end)
        if group_start < 0 or group_end <= group_start or group_end > num_layers:
            raise RuntimeError(
                f"same_layer_batch invalid group [{group_start}, {group_end}) "
                f"layers={num_layers}"
            )

        d_output = self._strip_lifecycle_fields(
            subset_scheduler_output(
                scheduler_output,
                d_req_ids,
                layered_plan=None,
                include_one_time_updates=False,
            )
        )
        mixed_output = self._strip_lifecycle_fields(scheduler_output)
        last_group = group_end == num_layers

        if group_start > 0:
            d_head_state, d_head_out = self._run_layered_prefill_subbatch(
                d_output,
                layered_plan,
                layer_start=0,
                layer_end=group_start,
                expect_final_layer=False,
                force_embed=True,
                skip_store=True,
                force_hidden=True,
                skip_kv_pre_forward=True,
            )
            store_req_frontiers(
                self.layered_prefill_state,
                list(d_head_state.input_batch.req_ids[: d_head_state.input_batch.num_reqs]),
                np.array(d_head_state.input_batch.query_start_loc_np, copy=True),
                d_head_out.hidden_states,
                d_head_out.residual,
                layered_plan.group_id,
            )
            torch.npu.synchronize()

        with self._layered_p_buffers():
            mixed_state, mixed_out = self._run_layered_prefill_subbatch(
                mixed_output,
                layered_plan,
                layer_start=group_start,
                layer_end=group_end,
                expect_final_layer=last_group,
                force_embed=(group_start == 0),
                force_hidden=True,
                store_req_ids=None if last_group else p_req_ids,
                skip_store=last_group,
            )
        torch.npu.synchronize()
        for req_id in d_req_ids:
            self.layered_prefill_state.clear(req_id)
        if not last_group:
            for req_id in p_req_ids:
                frontier = self.layered_prefill_state.get(req_id)
                expected = int(scheduler_output.num_scheduled_tokens[req_id])
                if frontier is None or int(frontier.query_len) != expected:
                    got = None if frontier is None else frontier.query_len
                    raise RuntimeError(
                        f"same_layer_batch P frontier {req_id} query_len="
                        f"{got} expected {expected}"
                    )

        mixed_req_ids = list(
            mixed_state.input_batch.req_ids[: mixed_state.input_batch.num_reqs]
        )
        mixed_loc = np.array(mixed_state.input_batch.query_start_loc_np, copy=True)
        if last_group:
            mixed_state = mixed_state._replace(
                hidden_states=mixed_out.hidden_states.clone()
            )
            self.execute_model_state = LayeredV2ExecuteModelState(
                scheduler_output=scheduler_output,
                d_state=None,
                p_state=mixed_state,
                sample_p=True,
                fused_mixed=True,
            )
            return None

        d_hidden, d_residual = slice_req_activations(
            mixed_req_ids,
            d_req_ids,
            mixed_loc,
            mixed_out.hidden_states,
            mixed_out.residual,
        )
        d_tail_state, d_tail_out = self._run_layered_prefill_subbatch(
            d_output,
            layered_plan,
            layer_start=group_end,
            layer_end=num_layers,
            expect_final_layer=True,
            explicit_frontier=(d_hidden, d_residual),
            skip_store=True,
            force_hidden=True,
            skip_kv_pre_forward=True,
        )
        from types import SimpleNamespace

        p_state = ExecuteModelState(
            input_batch=SimpleNamespace(
                req_ids=list(p_req_ids),
                num_reqs=len(p_req_ids),
            ),
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=None,
            aux_hidden_states=None,
            finished_req_ids=scheduler_output.finished_req_ids,
        )
        d_tail_state = d_tail_state._replace(hidden_states=d_tail_out.hidden_states)
        self.execute_model_state = LayeredV2ExecuteModelState(
            scheduler_output=scheduler_output,
            d_state=d_tail_state,
            p_state=p_state,
            sample_p=False,
        )
        return None

    @staticmethod
    def _strip_lifecycle_fields(scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """Clear one-shot lifecycle fields after they have already been applied."""
        from vllm.v1.core.sched.output import CachedRequestData
        from dataclasses import replace

        return replace(
            scheduler_output,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            preempted_req_ids=set(),
            free_encoder_mm_hashes=[],
            scheduled_encoder_input_stats=None,
            new_block_ids_to_zero=None,
            kv_cache_block_copies=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            ec_manager_metadata=None,
        )

    def reset_layered_prefill_counters(self) -> None:
        self.layered_prefill_counters = empty_layered_prefill_counters()

    def _layered_counters(self) -> dict:
        counters = getattr(self, "layered_prefill_counters", None)
        if counters is None:
            counters = empty_layered_prefill_counters()
            self.layered_prefill_counters = counters
        return counters

    def _record_layered_step(
        self,
        layered_plan,
        *,
        n_d: int,
        n_p: int,
        fused_mixed: bool = False,
        same_layer: bool = False,
    ) -> None:
        counters = self._layered_counters()
        counters["execute_steps"] += 1
        if fused_mixed:
            counters["fused_mixed_steps"] = int(counters.get("fused_mixed_steps") or 0) + 1
        if same_layer:
            counters["same_layer_steps"] = int(counters.get("same_layer_steps") or 0) + 1
        counters["groups"].append(
            {
                "group_id": int(layered_plan.group_id),
                "num_groups": int(layered_plan.num_groups),
                "group_start": int(layered_plan.group_start),
                "group_end": int(layered_plan.group_end),
                "is_final": bool(layered_plan.is_final_group),
                "is_sampling_step": bool(
                    getattr(layered_plan, "is_sampling_step", layered_plan.is_final_group)
                ),
                "is_final_chunk": bool(getattr(layered_plan, "is_final_chunk", True)),
                "n_d": int(n_d),
                "n_p": int(n_p),
                "fused_mixed": bool(fused_mixed),
                "same_layer": bool(same_layer),
                "cached_tokens": int(
                    (getattr(layered_plan, "cached_tokens", None) or {}).get(
                        layered_plan.prefill_req_ids[0], 0
                    )
                    if getattr(layered_plan, "prefill_req_ids", None)
                    else 0
                ),
            }
        )

    def _record_layered_pp_slot(self, slot: dict | None) -> None:
        if not slot:
            return
        self._layered_counters()["pp_slots"].append(slot)

    def _record_layered_activation(self, layered_plan, source: str) -> None:
        counters = self._layered_counters()
        pp = get_pp_group()
        counters["activation_sources"].append(
            {
                "group_id": int(layered_plan.group_id),
                "pp_rank": int(pp.rank_in_group),
                "owner": int(self._layered_pp_group_owner(layered_plan)),
                "source": source,
            }
        )
        if source == "transport_frontier":
            counters["transport_frontier_steps"] += 1

    def layered_prefill_snapshot(self) -> dict:
        """JSON-safe worker state for smoke probes (via collective_rpc)."""
        parallel = getattr(self, "parallel_config", None)
        counters = self._layered_counters()
        try:
            pp_rank = int(get_pp_group().rank_in_group)
        except Exception:
            pp_rank = -1
        return {
            "runner_class": f"{type(self).__module__}.{type(self).__name__}",
            "pp_size": int(getattr(parallel, "pipeline_parallel_size", -1)),
            "tp_size": int(getattr(parallel, "tensor_parallel_size", -1)),
            "pp_rank": pp_rank,
            "is_first_pp_rank": bool(getattr(self, "is_first_pp_rank", False)),
            "is_last_pp_rank": bool(getattr(self, "is_last_pp_rank", False)),
            "has_pp_handler": getattr(self, "pp_handler", None) is not None,
            "layered_enabled": bool(getattr(self, "_layered_prefill_enabled", False)),
            "fuse_mixed_batch": bool(getattr(self, "_fuse_mixed_batch", False)),
            "same_layer_batch": bool(getattr(self, "_same_layer_batch", False)),
            "layered_ready": bool(getattr(self, "_layered_prefill_v2_ready", False)),
            "layered_adapter": (
                getattr(self, "layered_prefill_model_adapter", None) is not None
            ),
            "layered_buffers": (
                getattr(self, "_layered_input_buffers", None) is not None
            ),
            "execute_steps": int(counters["execute_steps"]),
            "fused_mixed_steps": int(counters.get("fused_mixed_steps") or 0),
            "same_layer_steps": int(counters.get("same_layer_steps") or 0),
            "transport_frontier_steps": int(
                counters.get("transport_frontier_steps") or 0
            ),
            "skip_relay_stages": bool(getattr(self, "_skip_relay_stages", False)),
            "relay_empty_steps": int(counters.get("relay_empty_steps") or 0),
            "relay_forward_steps": int(counters.get("relay_forward_steps") or 0),
            "groups": list(counters["groups"]),
            "pp_slots": list(counters["pp_slots"]),
            "activation_sources": list(counters.get("activation_sources") or []),
        }

    def _apply_scheduler_lifecycle(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Run the one-shot request lifecycle that opens execute_model."""
        self.update_pp_decode_requests()
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()

    def _sync_ascend_num_computed_tokens_cpu(self, req_ids: list[str]) -> None:
        """Refresh Ascend's CPU num_computed mirror after lifecycle.

        ``update_requests`` only writes ``num_computed_tokens_np``.  Ascend's
        ``_update_seq_lens_cpu`` copies np→cpu for cached rows; when those rows
        are stripped for the D/P sub-batches, a stale cpu value makes
        ``seq_lens = cpu + scheduled`` one step behind and decode collapses.
        """
        for req_id in req_ids:
            req_index = self.req_states.req_id_to_index[req_id]
            self.req_states.num_computed_tokens_cpu[req_index] = int(
                self.req_states.num_computed_tokens_np[req_index]
            )

    def _layered_relay_stage_action(self, layered_plan) -> str | None:
        """How this stage should relay the active layer group, if at all.

        Returns ``"empty"`` for a stage ahead of the group owner,
        ``"forward"`` for one behind it, and None when the stage has real
        work.  A relay stage computes no layer (the adapter's local range is
        empty) and writes no KV, because the layers it owns belong to groups
        that already ran.

        Real work means owning the active group, or owning the *next* one:
        ``store_req_frontiers`` tags its result ``group_id + 1``, so the
        frontier has to be left on ``owner_of(group_id + 1)``.  Everyone else
        keeps nothing anyone will read -- a stage ahead of the owner stays
        ahead for the rest of the chunk, since the owner index never
        decreases.

        Four carve-outs:

        * The sampling step.  There the last rank broadcasts sampled tokens
          and every other rank has to be inside ``sample_tokens`` to receive
          them, or its request state diverges.  On every other step
          ``LayeredPPHandlerCapture.flush`` drops a P-only payload without
          broadcasting, so there is nothing to miss.
        * The final layer group, even when it does not sample (the last group
          of a non-final token chunk).  That step clears the request's stored
          frontier on every rank, and a rank that skipped the clear would
          still be holding one when the next chunk opens at group 0 -- which
          ``_layered_prefill_activation_source`` rejects outright.
        * ``dp_size > 1``.  ``dispatch_cg_and_sync_dp`` is a collective and
          must not be desynchronised.  Layered already rejects DP > 1; this
          keeps the reason local to the skip.
        * A KV connector.  Without one the executor collects a
          ``ModelRunnerOutput`` from the last PP stage only
          (``_get_output_rank``).  With one it aggregates every rank, and the
          skipped ``kv_connector.post_forward`` would be missed.  Layered
          already rejects connectors; same reasoning, kept local.
        """
        if not self._skip_relay_stages:
            return None
        if self.parallel_config.pipeline_parallel_size <= 1:
            return None
        if self.dp_size > 1:
            return None
        if self.vllm_config.kv_transfer_config is not None:
            return None
        if getattr(
            layered_plan, "is_sampling_step", layered_plan.is_final_group
        ):
            return None
        if layered_plan.is_final_group:
            return None
        rank = get_pp_group().rank_in_group
        owner = self._layered_pp_group_owner(layered_plan)
        if rank == owner:
            return None
        if rank == self._layered_next_group_owner(layered_plan):
            return None
        return "empty" if rank < owner else "forward"

    def _layered_next_group_owner(self, layered_plan) -> int | None:
        """PP rank owning the group after this one, or None if it is last.

        ``store_req_frontiers`` stamps its result with ``group_id + 1``, so
        that rank is the one stage besides the current owner which has to run
        the step for its stored frontier.
        """
        num_groups = int(layered_plan.num_groups)
        next_group = int(layered_plan.group_id) + 1
        if next_group >= num_groups:
            return None
        num_layers = self._num_hidden_layers()
        pp_size = self.parallel_config.pipeline_parallel_size
        layer_range = make_pp_aligned_layer_group_ranges(
            num_layers, num_groups, pp_size
        )[next_group]
        for rank in range(pp_size):
            start, end = get_pp_indices(num_layers, rank, pp_size)
            if start <= layer_range.start and layer_range.end <= end:
                return rank
        return None

    def _execute_layered_relay_step(
        self,
        scheduler_output: SchedulerOutput,
        action: str,
        received,
        *,
        fused: bool,
    ):
        """Hand the activation on without touching the layer group.

        ``sample_tokens`` still owes ``scheduler.update_from_output`` a row
        per scheduled request even though nothing sampled, so stash that
        output rather than leaving the state empty.
        """
        counters = self._layered_counters()
        key = f"relay_{action}_steps"
        counters[key] = int(counters.get(key) or 0) + 1
        logger.info(
            "Layered Prefill V2 relay action=%s pp_rank=%s empty=%s forward=%s",
            action,
            get_pp_group().rank_in_group,
            counters["relay_empty_steps"],
            counters["relay_forward_steps"],
        )
        self.execute_model_state = None
        req_ids = list(scheduler_output.num_scheduled_tokens)
        self._layered_relay_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={
                req_id: index for index, req_id in enumerate(req_ids)
            },
            sampled_token_ids=[[] for _ in req_ids],
        )
        if self.is_last_pp_rank:
            return None
        if action == "empty":
            return self._layered_pre_owner_placeholder(with_row_metadata=not fused)
        if received is None:
            raise RuntimeError("Layered relay stage has no activation to pass on")
        # Accessing ``.tensors`` on an AsyncIntermediateTensors waits for the
        # recv, so the worker's isend cannot race the incoming transfer.
        return received

    def _layered_pre_owner_placeholder(
        self, *, with_row_metadata: bool = False
    ) -> IntermediateTensors:
        """Zero-row payload standing in for a skipped layer group.

        Same contract as the sliced payload in ``_layered_fused_pp_payload``:
        no stage reads a pre-owner activation, and an empty tensor makes both
        ends of the ring skip the transfer while still exchanging metadata.

        Off the fused path the receiver runs its payload through
        ``_split_layered_pp_intermediate``, which insists on the D/P row
        counts even when it goes on to discard the rows -- the owner of a
        group past the first restores its own stored frontier instead
        (``activation_sources`` records it as ``frontier``).  Carry zeroed
        counts there so the unpack has something to read.
        """
        adapter = self.layered_prefill_model_adapter
        assert adapter is not None
        hidden, residual = adapter.make_transport_frontier(
            0, self.model_config.dtype, self.device
        )
        packed = adapter.to_intermediate_tensors(
            LayeredForwardOutput(hidden, residual, False)
        )
        if with_row_metadata:
            packed.tensors["layered_pp_d_rows"] = torch.tensor(0, dtype=torch.int64)
            packed.tensors["layered_pp_p_rows"] = torch.tensor(0, dtype=torch.int64)
        return packed

    def _layered_fused_pp_payload(
        self, layered_plan, output
    ) -> IntermediateTensors:
        """Pack a fused layer-group result for the next PP stage.

        Stages before the group owner hold an activation that nobody reads:
        the owner restores its own stored frontier and every earlier stage
        rebuilds its own ``make_transport_frontier`` placeholder.  Send zero
        rows there.  The receiver sizes its buffer from the sender's tensor
        metadata and skips the transfer entirely for an empty tensor, so the
        send/recv pair stays matched without moving a full-length
        activation -- 268 MB per hop at a 32k prompt.

        Only the fused payload is all-P.  The serial D->P path also carries
        decode rows that every stage really does need, so it keeps sending
        the full payload.
        """
        packed = self.layered_prefill_model_adapter.to_intermediate_tensors(
            output
        )
        if get_pp_group().rank_in_group >= self._layered_pp_group_owner(
            layered_plan
        ):
            return packed
        return IntermediateTensors(
            {key: value[:0] for key, value in packed.tensors.items()}
        )

    def _layered_pp_group_owner(self, plan) -> int:
        pp = get_pp_group()
        configs = (
            getattr(self.model_config, "hf_text_config", None),
            getattr(self.model_config, "hf_config", None),
            self.model_config,
        )
        num_layers = int(plan.group_end)
        for config in configs:
            raw_num_layers = getattr(config, "num_hidden_layers", None)
            if isinstance(raw_num_layers, int):
                num_layers = int(raw_num_layers)
                break
        for rank in range(pp.world_size):
            start, end = get_pp_indices(num_layers, rank, pp.world_size)
            if start <= plan.group_start and plan.group_end <= end:
                return rank
        raise RuntimeError(
            f"Layered group [{plan.group_start}, {plan.group_end}) is not "
            "aligned with the PP layer partition"
        )

    @staticmethod
    def _layered_pp_row_count(value, key: str) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            row_count = value
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            row_count = int(value.item())
        else:
            raise RuntimeError(
                f"Layered PP metadata {key} must be an integer or scalar tensor"
            )
        if row_count < 0:
            raise RuntimeError(f"Layered PP metadata {key} cannot be negative")
        return row_count

    @staticmethod
    def _clone_intermediate_tensors(
        result: IntermediateTensors,
    ) -> IntermediateTensors:
        cloned = IntermediateTensors(
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in result.tensors.items()
            }
        )
        cloned.kv_connector_output = getattr(result, "kv_connector_output", None)
        return cloned

    @staticmethod
    def _split_layered_pp_intermediate(
        intermediate_tensors: IntermediateTensors | None,
    ) -> tuple[IntermediateTensors | None, IntermediateTensors | None]:
        if intermediate_tensors is None:
            return None, None
        tensors = intermediate_tensors.tensors
        if "layered_pp_d_rows" not in tensors or "layered_pp_p_rows" not in tensors:
            raise RuntimeError(
                "Layered PP intermediate tensors are missing D/P row metadata"
            )
        d_rows = NPUModelRunner._layered_pp_row_count(
            tensors["layered_pp_d_rows"], "layered_pp_d_rows"
        )
        p_rows = NPUModelRunner._layered_pp_row_count(
            tensors["layered_pp_p_rows"], "layered_pp_p_rows"
        )
        tensor_items = [
            (key, value)
            for key, value in tensors.items()
            if key not in ("layered_pp_d_rows", "layered_pp_p_rows")
            and isinstance(value, torch.Tensor)
        ]
        if not tensor_items:
            raise RuntimeError("Layered PP intermediate tensors contain no tensors")
        for key, value in tensor_items:
            if value.ndim == 0 or d_rows + p_rows > value.shape[0]:
                raise RuntimeError(
                    f"Layered PP metadata exceeds {key} row dimension: "
                    f"{d_rows}+{p_rows}>{value.shape[0] if value.ndim else 0}"
                )
        d_tensors = {key: value[:d_rows] for key, value in tensor_items}
        p_tensors = {
            key: value[d_rows : d_rows + p_rows] for key, value in tensor_items
        }
        return IntermediateTensors(d_tensors), IntermediateTensors(p_tensors)

    @staticmethod
    def _combine_layered_pp_intermediate(
        d_intermediate: IntermediateTensors | None,
        p_intermediate: IntermediateTensors | None,
    ) -> IntermediateTensors:
        if d_intermediate is None and p_intermediate is None:
            raise RuntimeError("Layered PP pack requires D or P intermediates")
        d_tensors = d_intermediate.tensors if d_intermediate is not None else {}
        p_tensors = p_intermediate.tensors if p_intermediate is not None else {}
        keys = dict.fromkeys((*d_tensors, *p_tensors))
        tensors: dict = {}
        for key in keys:
            d_value = d_tensors.get(key)
            p_value = p_tensors.get(key)
            if isinstance(d_value, torch.Tensor) and isinstance(p_value, torch.Tensor):
                tensors[key] = torch.cat((d_value, p_value), dim=0)
            elif isinstance(d_value, torch.Tensor):
                tensors[key] = d_value
            elif isinstance(p_value, torch.Tensor):
                tensors[key] = p_value
        d_rows = (
            0
            if d_intermediate is None
            else next(
                (
                    value.shape[0]
                    for value in d_tensors.values()
                    if isinstance(value, torch.Tensor)
                ),
                0,
            )
        )
        p_rows = (
            0
            if p_intermediate is None
            else next(
                (
                    value.shape[0]
                    for value in p_tensors.values()
                    if isinstance(value, torch.Tensor)
                ),
                0,
            )
        )
        if not tensors:
            raise RuntimeError("Layered PP intermediate tensors contain no tensors")
        tensors["layered_pp_d_rows"] = torch.tensor(d_rows, dtype=torch.int64)
        tensors["layered_pp_p_rows"] = torch.tensor(p_rows, dtype=torch.int64)
        return IntermediateTensors(tensors)

    def _prepare_layered_p_activation(
        self,
        layered_plan,
        *,
        req_ids: list[str] | None = None,
        num_tokens_padded: int,
        inputs_embeds,
        intermediate_tensors: IntermediateTensors | None,
    ):
        """Pick frontier / embed / PP transport inputs for the P (or mixed) batch."""
        layered_adapter = self.layered_prefill_model_adapter
        assert layered_adapter is not None
        if req_ids is None:
            req_ids = list(layered_plan.prefill_req_ids)
        if not req_ids:
            raise RuntimeError("Layered Prefill activation has no request ids")
        pp = get_pp_group()
        owner = self._layered_pp_group_owner(layered_plan)
        owner_has_frontier = layered_plan.group_id > 0 and pp.rank_in_group == owner
        if owner_has_frontier:
            if len(req_ids) > 1:
                frontier_tuple = concat_req_frontiers(
                    self.layered_prefill_state,
                    req_ids,
                    expected_group_id=layered_plan.group_id,
                    num_tokens_padded=num_tokens_padded,
                )
                return req_ids, frontier_tuple, None, None, "frontier_mixed"
            req_id = req_ids[0]
            frontier = self.layered_prefill_state.get(req_id)
            if frontier is None:
                raise RuntimeError(
                    f"Missing layered activation frontier for request {req_id}"
                )
            if frontier.group_id != layered_plan.group_id:
                raise RuntimeError(
                    f"Layered frontier group mismatch for {req_id}: expected "
                    f"{layered_plan.group_id}, got {frontier.group_id}"
                )
            if os.environ.get("VLLM_LAYERED_ROW_TRACE") == "1":
                logger.info(
                    "layered_row restore_single req=%s group=%s "
                    "stored_hidden=%s stored_residual=%s query_len=%s "
                    "num_tokens_padded=%s",
                    req_id,
                    layered_plan.group_id,
                    tuple(frontier.hidden_states.shape),
                    None
                    if frontier.residual is None
                    else tuple(frontier.residual.shape),
                    frontier.query_len,
                    num_tokens_padded,
                )
            return (
                req_ids,
                (frontier.hidden_states, frontier.residual),
                None,
                None,
                "frontier",
            )

        req_id = req_ids[0]
        frontier = self.layered_prefill_state.get(req_id)
        if frontier is not None and pp.rank_in_group == owner:
            raise RuntimeError(
                f"Unexpected layered frontier for group 0 request {req_id}"
            )
        frontier_tuple = None
        if layered_plan.group_id == 0 and pp.rank_in_group == 0:
            initial_inputs_embeds = inputs_embeds
            source = "embed"
        elif pp.rank_in_group < owner and layered_plan.group_id > 0:
            # D5: ranks before the group owner still participate in the
            # PP send/recv chain.  A dummy activation keeps NCCL matched
            # without consuming a leftover local frontier.
            frontier_tuple = layered_adapter.make_transport_frontier(
                num_tokens_padded,
                self.model_config.dtype,
                self.device,
            )
            initial_inputs_embeds = None
            source = "transport_frontier"
        else:
            initial_inputs_embeds = None
            source = "pp_recv" if pp.rank_in_group > owner else "none"
        receives_pp_activation = pp.rank_in_group > owner
        if receives_pp_activation and intermediate_tensors is None:
            raise RuntimeError("Layered PP stage did not receive P activation")
        return (
            req_ids,
            frontier_tuple,
            initial_inputs_embeds,
            intermediate_tensors if receives_pp_activation else None,
            source,
        )

    @staticmethod
    def _layered_prefill_moe_layer_offset(
        all_moe_layers: list[str], layer_start: int
    ) -> int:
        """Return the MoE custom-op layer offset for a Transformer layer boundary."""
        return sum(
            extract_layer_index(layer_name) < layer_start
            for layer_name in all_moe_layers
        )

    def _set_layered_prefill_moe_layer_offset(self, layer_start: int) -> None:
        forward_context = get_forward_context()
        all_moe_layers = forward_context.all_moe_layers
        # Empty list means the compile inventory was never filled; treat as unset
        # so MoE ops fall back to baking layer names rather than indexing [].
        if not all_moe_layers:
            return
        offset = self._layered_prefill_moe_layer_offset(all_moe_layers, layer_start)
        forward_context.moe_layer_index = offset
        # Optional cross-rank / debug ledger (V5). Enable with
        # VLLM_ASCEND_LAYERED_MOE_LOG=1.
        if os.environ.get("VLLM_ASCEND_LAYERED_MOE_LOG") == "1":
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX

            logger.info(
                "layered_moe plan group_start=%s moe_offset=%s/%s "
                "comm=%s num_tokens=%s",
                layer_start,
                offset,
                len(all_moe_layers),
                getattr(_EXTRA_CTX, "moe_comm_type", None),
                getattr(_EXTRA_CTX, "num_tokens", None),
            )

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        """Pad query tokens to a TP multiple when DSA-CP or native SP is on.

        Same contract as V1 ``NPUModelRunner._pad_for_sequence_parallelism``:
        scheduler bookkeeping stays on the logical length.
        """
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if enable_sp(self.vllm_config) or enable_dsa_cp():
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _run_layered_prefill_subbatch(
        self,
        scheduler_output: SchedulerOutput,
        layered_plan,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        layer_start: int | None = None,
        layer_end: int | None = None,
        expect_final_layer: bool | None = None,
        force_embed: bool = False,
        explicit_frontier: tuple | None = None,
        store_req_ids: list[str] | None = None,
        skip_store: bool = False,
        force_hidden: bool = False,
        skip_kv_pre_forward: bool = False,
    ) -> tuple[ExecuteModelState, LayeredForwardOutput]:
        """Eager layer-range forward: prepare_inputs → prepare_attn → adapter."""
        if scheduler_output.total_num_scheduled_tokens == 0:
            raise RuntimeError("Layered Prefill P sub-batch has zero tokens")

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)

        # Prefill layer groups stay eager (migration plan §2 / V4).
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            need_eager=True,
            num_active_loras=0,
        )
        if batch_desc.cg_mode != CUDAGraphMode.NONE:
            raise RuntimeError(
                "Layered Prefill P sub-batch must stay eager "
                f"(got cudagraph mode {batch_desc.cg_mode})"
            )
        logger.info_once("Layered Prefill subbatch selected eager execution")
        if batch_desc.num_tokens == 0:
            raise RuntimeError("Layered Prefill P sub-batch dispatched zero tokens")
        # Match V1 / origin dsa_cp: pad the physical eager batch to a TP
        # multiple while scheduler_output keeps the logical prompt length.
        # KV/attention use the logical count; MoE collectives use the padded
        # width. Pad rows get slot_id=-1 (compute_slot_mappings) and zeroed
        # positions.
        padded_toks = self._pad_for_sequence_parallelism(num_toks)
        if padded_toks != batch_desc.num_tokens:
            batch_desc = replace(batch_desc, num_tokens=padded_toks)

        if not vllm_version_is("0.27.1"):
            raise NotImplementedError(
                "Layered Prefill V2 P path currently requires the 0.27.1-shaped "
                "prepare_inputs API (set VLLM_VERSION=0.27.1 on this tree)"
            )

        input_batch = self.prepare_inputs(scheduler_output, batch_desc)
        if padded_toks != input_batch.num_tokens_after_padding:
            raise RuntimeError(
                "Layered Prefill P pad width mismatch: batch_desc="
                f"{padded_toks}, input_batch={input_batch.num_tokens_after_padding}"
            )
        if input_batch.num_tokens_after_padding > input_batch.num_tokens:
            pad_from = input_batch.num_tokens
            pad_to = input_batch.num_tokens_after_padding
            self.input_buffers.positions[pad_from:pad_to].fill_(0)
        block_tables, slot_mappings = self.prepare_attn(input_batch)
        self.model_state.preprocess_state(
            input_batch,
            block_tables,
            self.kv_cache_config,
            self.req_states.num_computed_tokens.gpu,
        )

        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            batch_desc.cg_mode,
            block_tables,
            slot_mappings,
            self.attn_groups,
            self.kv_cache_config,
            for_capture=False,
            num_input_tokens=padded_toks if padded_toks > num_toks else None,
        )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        positions = input_batch.positions
        num_tokens_padded = input_batch.num_tokens_after_padding

        batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens_padded,
            has_lora=False,
            num_active_loras=0,
        )
        layered_adapter = self.layered_prefill_model_adapter
        assert layered_adapter is not None
        num_reqs = input_batch.num_reqs
        batch_req_ids = list(input_batch.req_ids[:num_reqs])
        run_start = (
            int(layered_plan.group_start) if layer_start is None else int(layer_start)
        )
        run_end = int(layered_plan.group_end) if layer_end is None else int(layer_end)
        run_final = (
            bool(layered_plan.is_final_group)
            if expect_final_layer is None
            else bool(expect_final_layer)
        )
        if explicit_frontier is not None:
            frontier_tuple = pad_activation_rows(
                explicit_frontier[0],
                explicit_frontier[1],
                num_tokens_padded,
            )
            initial_inputs_embeds = None
            p_pp_intermediate = None
            activation_source = "explicit"
        elif force_embed:
            frontier_tuple = None
            initial_inputs_embeds = inputs_embeds
            p_pp_intermediate = None
            activation_source = "embed"
        else:
            (
                _act_req_ids,
                frontier_tuple,
                initial_inputs_embeds,
                p_pp_intermediate,
                activation_source,
            ) = self._prepare_layered_p_activation(
                layered_plan,
                req_ids=batch_req_ids,
                num_tokens_padded=num_tokens_padded,
                inputs_embeds=inputs_embeds,
                intermediate_tensors=intermediate_tensors,
            )
        self._record_layered_activation(layered_plan, activation_source)
        if os.environ.get("VLLM_LAYERED_ROW_TRACE") == "1":
            qsl = input_batch.query_start_loc_np
            logger.info(
                "layered_row step reqs=%s group=%s/%s layers=[%s,%s) "
                "logical_query=%s padded_rows=%s source=%s qsl=%s "
                "frontier_hidden=%s frontier_residual=%s input_ids=%s positions=%s",
                batch_req_ids,
                layered_plan.group_id,
                layered_plan.num_groups,
                run_start,
                run_end,
                num_toks,
                num_tokens_padded,
                activation_source,
                [int(qsl[i]) for i in range(num_reqs + 1)],
                None if frontier_tuple is None else tuple(frontier_tuple[0].shape),
                None
                if frontier_tuple is None or frontier_tuple[1] is None
                else tuple(frontier_tuple[1].shape),
                tuple(input_ids.shape),
                tuple(positions.shape),
            )

        with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_toks,
            aclgraph_runtime_mode=batch_desc.cg_mode,
            batch_descriptor=batch_descriptor,
            model_instance=self.model,
            skip_compiled=True,
            moe_comm_token_count=num_toks,
        ):
            ctx = get_forward_context()
            ctx.slot_mapping = slot_mappings_by_layer
            ctx.is_padding = input_batch.is_padding
            self._set_layered_prefill_moe_layer_offset(run_start)
            if not skip_kv_pre_forward:
                self.kv_connector.pre_forward(scheduler_output)

            layered_output = layered_adapter.forward(
                input_ids=input_ids,
                positions=positions[:num_tokens_padded],
                layer_start=run_start,
                layer_end=run_end,
                frontier=frontier_tuple,
                inputs_embeds=initial_inputs_embeds,
                intermediate_tensors=p_pp_intermediate,
            )

        actual_rows = int(layered_output.hidden_states.shape[0])
        if actual_rows != num_tokens_padded:
            raise RuntimeError(
                "Layered model returned a hidden-state row count that "
                "does not match the P query batch: "
                f"actual_rows={actual_rows} expected_rows={num_tokens_padded} "
                f"actual_hidden={tuple(layered_output.hidden_states.shape)} "
                f"actual_residual="
                f"{None if layered_output.residual is None else tuple(layered_output.residual.shape)} "
                f"group={layered_plan.group_id}/{layered_plan.num_groups} "
                f"layers=[{run_start},{run_end}) logical_query={num_toks} "
                f"reqs={batch_req_ids} source={activation_source}"
            )
        if layered_output.is_final_layer != run_final:
            raise RuntimeError(
                "Layered model final-layer status does not match the plan"
            )

        if run_final:
            for req_id in batch_req_ids:
                self.layered_prefill_state.clear(req_id)
            hidden_states = layered_output.hidden_states
        elif skip_store:
            hidden_states = layered_output.hidden_states if force_hidden else None
        else:
            store_ids = batch_req_ids if store_req_ids is None else list(store_req_ids)
            if store_ids:
                store_req_frontiers(
                    self.layered_prefill_state,
                    store_ids,
                    np.array(input_batch.query_start_loc_np, copy=True),
                    layered_output.hidden_states,
                    layered_output.residual,
                    layered_plan.group_id + 1,
                    batch_req_ids=batch_req_ids,
                    keep_ids=store_ids,
                )
                if os.environ.get("VLLM_LAYERED_ROW_TRACE") == "1":
                    for req_id in store_ids:
                        saved = self.layered_prefill_state.get(req_id)
                        logger.info(
                            "layered_row store req=%s next_group=%s "
                            "src_hidden=%s src_residual=%s saved_hidden=%s "
                            "saved_residual=%s query_len=%s",
                            req_id,
                            layered_plan.group_id + 1,
                            tuple(layered_output.hidden_states.shape),
                            None
                            if layered_output.residual is None
                            else tuple(layered_output.residual.shape),
                            None if saved is None else tuple(saved.hidden_states.shape),
                            None
                            if saved is None or saved.residual is None
                            else tuple(saved.residual.shape),
                            None if saved is None else saved.query_len,
                        )
            hidden_states = layered_output.hidden_states if force_hidden else None

        return (
            ExecuteModelState(
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings_by_layer=slot_mappings_by_layer,
                hidden_states=hidden_states,
                aux_hidden_states=None,
                finished_req_ids=scheduler_output.finished_req_ids,
            ),
            layered_output,
        )

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        relay_output = self._layered_relay_output
        if relay_output is not None:
            # This stage only passed the activation on.  Nothing sampled, but
            # every scheduled request still needs a row in the output.
            self._layered_relay_output = None
            return relay_output
        state = self.execute_model_state
        if isinstance(state, LayeredV2ExecuteModelState):
            return self._sample_layered_v2(grammar_output, state)
        return super().sample_tokens(grammar_output)

    def _sample_layered_v2(self, grammar_output, state: LayeredV2ExecuteModelState):
        outputs: list[ModelRunnerOutput] = []
        p_req_ids = (
            list(state.p_state.input_batch.req_ids) if state.p_state is not None else []
        )
        with self._layered_pp_handler_capture() as capture:
            try:
                if state.d_state is not None:
                    self.execute_model_state = state.d_state
                    active_grammar = grammar_output
                    if grammar_output is not None and not any(
                        req_id in grammar_output.structured_output_request_ids
                        for req_id in state.d_state.input_batch.req_ids
                    ):
                        active_grammar = None
                    output = super().sample_tokens(active_grammar)
                    if isinstance(output, AsyncOutput):
                        output = output.get_output()
                    if output is None:
                        output = EMPTY_MODEL_RUNNER_OUTPUT
                    if not isinstance(output, ModelRunnerOutput):
                        raise RuntimeError(
                            "Layered Prefill D sampling returned PP tensors"
                        )
                    outputs.append(output)

                if state.p_state is not None:
                    if state.sample_p:
                        # Last rank samples; non-last receives.  Both go through
                        # sample_tokens so token progress and the deferred
                        # PPHandler slot stay in lockstep.  Capture merges D ∪ P
                        # into one broadcast/receive (V6).
                        # fused_mixed: p_state is the combined D+P batch.
                        self.execute_model_state = state.p_state
                        output = super().sample_tokens(None)
                        if isinstance(output, AsyncOutput):
                            output = output.get_output()
                        if output is None:
                            output = EMPTY_MODEL_RUNNER_OUTPUT
                        if not isinstance(output, ModelRunnerOutput):
                            raise RuntimeError(
                                "Layered Prefill P sampling returned PP tensors"
                            )
                        outputs.append(output)
                    else:
                        # Non-sampling step (intermediate layer group, or the
                        # last group of a non-final prompt chunk): skip sampler,
                        # token-progress append, and the PPHandler slot.  Still
                        # emit req_ids so scheduler.update_from_output can find
                        # every scheduled request (empty sampled tokens).
                        # fused_mixed includes D rows here — they also wait.
                        finished = state.p_state.finished_req_ids
                        self.execute_model_state = None
                        kv_out = self.kv_connector.post_forward(finished)
                        outputs.append(
                            ModelRunnerOutput(
                                req_ids=p_req_ids,
                                req_id_to_index={
                                    req_id: i for i, req_id in enumerate(p_req_ids)
                                },
                                sampled_token_ids=[[] for _ in p_req_ids],
                                kv_connector_output=kv_out,
                            )
                        )
            finally:
                self.execute_model_state = None
            if capture is not None:
                slot = capture.flush(
                    sample_p=state.sample_p, p_req_ids=set(p_req_ids)
                )
                self._record_layered_pp_slot(slot)

        if not outputs:
            return EMPTY_MODEL_RUNNER_OUTPUT
        if len(outputs) == 1:
            return outputs[0]
        return self._merge_layered_v2_outputs(state.scheduler_output, outputs)

    @staticmethod
    def _merge_layered_v2_outputs(
        scheduler_output: SchedulerOutput,
        outputs: list[ModelRunnerOutput],
    ) -> ModelRunnerOutput:
        by_req_id: dict[str, tuple[ModelRunnerOutput, int]] = {}
        for output in outputs:
            for index, req_id in enumerate(output.req_ids):
                by_req_id[req_id] = (output, index)
        req_ids = list(scheduler_output.num_scheduled_tokens)
        sampled_token_ids = [
            by_req_id[req_id][0].sampled_token_ids[by_req_id[req_id][1]]
            if req_id in by_req_id
            else []
            for req_id in req_ids
        ]
        prompt_logprobs: dict = {}
        num_nans: dict = {}
        for output in outputs:
            prompt_logprobs.update(output.prompt_logprobs_dict)
            if output.num_nans_in_logits:
                num_nans.update(output.num_nans_in_logits)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=next((o.logprobs for o in outputs if o.logprobs is not None), None),
            prompt_logprobs_dict=prompt_logprobs,
            pooler_output=[],
            kv_connector_output=next(
                (o.kv_connector_output for o in outputs if o.kv_connector_output),
                None,
            ),
            ec_connector_output=next(
                (o.ec_connector_output for o in outputs if o.ec_connector_output),
                None,
            ),
            num_nans_in_logits=num_nans or None,
            cudagraph_stats=next(
                (o.cudagraph_stats for o in outputs if o.cudagraph_stats),
                None,
            ),
            routed_experts=None,
        )

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Override GPUModelRunner.profile_run for Ascend NPUs.
        When running moe models, we need an extra dummy run with mc2_tokens_capacity tokens to reserve
        necessary HCCL buffer for the MC2 operator before standard `profile_run`. Additionally, we set
        override_mrv2_in_profile_run to True to force moe load to be balanced when executing `profile_run`
        """
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        with override_mrv2_in_profile_run(True):
            if (
                mc2_tokens_capacity is not None
                and self.max_num_tokens > mc2_tokens_capacity
                and select_moe_comm_method(mc2_tokens_capacity, self.vllm_config)
                in {MoECommType.MC2, MoECommType.FUSED_MC2}
            ):
                self._dummy_run(mc2_tokens_capacity, skip_attn=True, skip_eplb=True, is_profile=True)
            super().profile_run()

    if vllm_version_is("0.27.1"):

        def prepare_inputs(
            self,
            scheduler_output: SchedulerOutput,
            batch_desc: BatchExecutionDescriptor,
        ) -> AscendInputBatch:
            """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
            npu attention backends need seq_lens_cpu to work.
            so we need to prepare seq_lens_cpu here.
            """
            num_tokens = scheduler_output.total_num_scheduled_tokens
            num_tokens_after_padding = batch_desc.num_tokens
            assert num_tokens > 0
            if num_tokens_after_padding > num_tokens:
                self.input_buffers.is_padding[:num_tokens].fill_(False)
                self.input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(
                    True
                )
            num_tokens_per_req = scheduler_output.num_scheduled_tokens
            num_reqs = len(num_tokens_per_req)

            req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)

            self._update_seq_lens_cpu(scheduler_output, req_ids)

            numtoks_iter = map(num_tokens_per_req.get, req_ids)
            num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)
            num_valid_tokens = num_scheduled_tokens
            if scheduler_output.scheduled_spec_decode_tokens:
                num_valid_tokens = np.array(
                    [
                        num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                        for num_tokens, i in zip(num_scheduled_tokens, req_ids)
                    ],
                    dtype=np.int32,
                )
            attn_state = build_attn_state(
                self.vllm_config,
                self.input_buffers.seq_lens_np,
                num_reqs,
                num_scheduled_tokens,
                num_valid_tokens,
            )
            idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
            idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
            idx_mapping_cpu = torch.from_numpy(idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_cpu, device=self.device)

            # Get the number of draft tokens for each request.
            draft_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_draft_tokens_per_req = None
            if not draft_tokens:
                # No draft token scheduled (common case).
                total_num_draft_tokens = 0
                total_num_logits = num_reqs
                cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
                cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
                expanded_idx_mapping = idx_mapping
                expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
            else:
                num_draft_tokens_per_req = np.fromiter(
                    (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
                total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
                total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
                num_logits = num_draft_tokens_per_req + num_bonus_tokens
                cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
                cu_num_logits_np[0] = 0
                np.cumsum(num_logits, out=cu_num_logits_np[1:])
                cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

                max_expand_len = self.decode_query_len
                expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                    idx_mapping, total_num_logits, cu_num_logits, max_expand_len
                )

            # Get query_start_loc.
            # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
            # See _pad_query_start_loc_for_fia.
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
            query_start_loc_np[0] = 0
            np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
            # Pad for full CUDA graph mode.
            # Some attention backends like FA3 require query_start_loc to be non-decreasing.
            query_start_loc_np[num_reqs + 1 :] = num_tokens

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                # This is only required for vllm-ascend.
                query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                    num_tokens_after_padding,
                    num_reqs_padded,
                    num_reqs,
                    query_start_loc_np,
                    batch_desc.cg_mode,
                    batch_desc.num_reqs,
                )

            async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

            query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
            query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
            prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
            num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[idx_mapping_np]
            is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
            batch_has_prefill = bool(np.any(is_prefilling_np))
            self.eplb.set_batch_phase(batch_has_prefill)

            # Get prefill tokens if any.
            if batch_has_prefill:
                prepare_prefill_inputs(
                    self.input_buffers.input_ids,
                    self.req_states.next_prefill_tokens,
                    idx_mapping,
                    query_start_loc,
                    self.req_states.all_token_ids.gpu,
                    self.req_states.prefill_len.gpu,
                    self.req_states.num_computed_tokens.gpu,
                )

            # Prepare positions and seq_lens.
            prepare_pos_seq_lens(
                idx_mapping,
                query_start_loc,
                self.req_states.num_computed_tokens.gpu,
                self.input_buffers.positions,
                self.input_buffers.seq_lens,
            )
            seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

            # Pad for full CUDA graph mode.  Also clear FIA dummy rows
            # [num_reqs, num_reqs_padded) so DSA-CP does not inherit leftover KV
            # lengths (see AscendDSACPMetadataBuilder._zero_graph_padding_rows).
            self.input_buffers.seq_lens_np[num_reqs:num_reqs_padded] = 0
            self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

            # Some input token ids are directly read from the last sampled tokens
            # and draft tokens. Also, get the logits indices to sample tokens from.
            logits_indices = combine_sampled_and_draft_tokens(
                self.input_buffers.input_ids,
                idx_mapping,
                self.req_states.last_sampled_tokens,
                query_start_loc,
                seq_lens,
                self.req_states.prefill_len.gpu,
                self.req_states.draft_tokens,
                cu_num_logits,
                total_num_logits,
                self.model_state.num_new_sampled_tokens_per_step,
            )

            # CPU upper bound on seq_lens (num_computed_tokens + num_scheduled_tokens).
            # Added by vLLM PR #40654 to avoid GPU->CPU sync for seq_lens.
            seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
            np.add(
                self.req_states.num_computed_tokens_np[idx_mapping_np],
                num_scheduled_tokens,
                out=seq_lens_cpu_upper_bound_np[:num_reqs],
            )
            seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)
            num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]

            max_seq_len_np = None
            if self.use_pp:
                # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
                max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

            prompt_lens = None
            if self.model_config.rswa_window is not None:
                # prompt_lens is only used in R-SWA case.
                prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

            input_batch = AscendInputBatch(
                req_ids=req_ids,
                num_reqs=num_reqs,
                num_reqs_after_padding=num_reqs_padded,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                expanded_idx_mapping=expanded_idx_mapping,
                expanded_local_pos=expanded_local_pos,
                num_scheduled_tokens=num_scheduled_tokens,
                num_tokens=num_tokens,
                num_tokens_after_padding=num_tokens_after_padding,
                num_draft_tokens=total_num_draft_tokens,
                num_draft_tokens_per_req=num_draft_tokens_per_req,
                query_start_loc=query_start_loc,
                query_start_loc_np=query_start_loc_np,
                seq_lens=seq_lens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
                is_prefilling_np=is_prefilling_np,
                num_computed_tokens_np=num_computed_tokens_np,
                prefill_len_np=prefill_len_np,
                num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
                max_seq_len_np=max_seq_len_np,
                input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
                positions=self.input_buffers.positions[:num_tokens_after_padding],
                is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
                logits_indices=logits_indices,
                cu_num_logits=cu_num_logits,
                cu_num_logits_np=cu_num_logits_np,
                has_structured_output_reqs=scheduler_output.has_structured_output_requests,
                # TODO: only populated for R-SWA (not supported yet).
                prompt_lens=prompt_lens,
                # extra attributes for ascend npus.
                seq_lens_np=self.input_buffers.seq_lens_np,
                attn_state=attn_state,
            )

            input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

            # For mla/sfa, update cos/sin. Here is for execute_model.
            update_cos_sin(input_batch.positions)

            return input_batch

    else:

        def prepare_inputs(  # type: ignore[misc]
            self,
            scheduler_output: SchedulerOutput,
            batch_req_state: BatchReqState,
            batch_desc: BatchExecutionDescriptor,
        ) -> AscendInputBatch:
            """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
            npu attention backends need seq_lens_cpu to work.
            so we need to prepare seq_lens_cpu here.
            """
            num_tokens = scheduler_output.total_num_scheduled_tokens
            num_tokens_after_padding = batch_desc.num_tokens
            assert num_tokens > 0
            if num_tokens_after_padding > num_tokens:
                self.input_buffers.is_padding[:num_tokens].fill_(False)
                self.input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(
                    True
                )
            num_tokens_per_req = scheduler_output.num_scheduled_tokens
            num_reqs = len(num_tokens_per_req)

            req_ids = sort_batch_req_ids(
                num_tokens_per_req,
                scheduler_output.scheduled_spec_decode_tokens,
                self.decode_query_len,
            )

            self._update_seq_lens_cpu(scheduler_output, req_ids)

            numtoks_iter = map(num_tokens_per_req.get, req_ids)
            num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)
            num_valid_tokens = num_scheduled_tokens
            if scheduler_output.scheduled_spec_decode_tokens:
                num_valid_tokens = np.array(
                    [
                        num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                        for num_tokens, i in zip(num_scheduled_tokens, req_ids)
                    ],
                    dtype=np.int32,
                )
            attn_state = build_attn_state(
                self.vllm_config,
                self.input_buffers.seq_lens_np,
                num_reqs,
                num_scheduled_tokens,
                num_valid_tokens,
            )
            idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
            idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
            idx_mapping_cpu = torch.from_numpy(idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_cpu, device=self.device)

            # Get the number of draft tokens for each request.
            draft_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_draft_tokens_per_req = None
            if not draft_tokens:
                # No draft token scheduled (common case).
                total_num_draft_tokens = 0
                total_num_logits = num_reqs
                cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
                cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
                expanded_idx_mapping = idx_mapping
                expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
            else:
                num_draft_tokens_per_req = np.fromiter(
                    (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
                total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
                total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
                num_logits = num_draft_tokens_per_req + num_bonus_tokens
                cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
                cu_num_logits_np[0] = 0
                np.cumsum(num_logits, out=cu_num_logits_np[1:])
                cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

                max_expand_len = self.decode_query_len
                expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                    idx_mapping, total_num_logits, cu_num_logits, max_expand_len
                )

            # Get query_start_loc.
            # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
            # See _pad_query_start_loc_for_fia.
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
            query_start_loc_np[0] = 0
            np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
            # Pad for full CUDA graph mode.
            # Some attention backends like FA3 require query_start_loc to be non-decreasing.
            query_start_loc_np[num_reqs + 1 :] = num_tokens

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                # This is only required for vllm-ascend.
                query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                    num_tokens_after_padding,
                    num_reqs_padded,
                    num_reqs,
                    query_start_loc_np,
                    batch_desc.cg_mode,
                    batch_desc.num_reqs,
                )

            async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

            query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
            query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
            prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
            num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[idx_mapping_np]
            is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
            batch_has_prefill = bool(np.any(is_prefilling_np))
            self.eplb.set_batch_phase(batch_has_prefill)

            # Get prefill tokens if any.
            if batch_has_prefill:
                prepare_prefill_inputs(
                    self.input_buffers.input_ids,
                    self.req_states.next_prefill_tokens,
                    idx_mapping,
                    query_start_loc,
                    self.req_states.all_token_ids.gpu,
                    self.req_states.prefill_len.gpu,
                    self.req_states.num_computed_tokens.gpu,
                )

            # Prepare positions and seq_lens.
            prepare_pos_seq_lens(
                idx_mapping,
                query_start_loc,
                self.req_states.num_computed_tokens.gpu,
                self.input_buffers.positions,
                self.input_buffers.seq_lens,
            )
            seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

            # Pad for full CUDA graph mode.  Also clear FIA dummy rows
            # [num_reqs, num_reqs_padded) so DSA-CP does not inherit leftover KV
            # lengths (see AscendDSACPMetadataBuilder._zero_graph_padding_rows).
            self.input_buffers.seq_lens_np[num_reqs:num_reqs_padded] = 0
            self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

            # Some input token ids are directly read from the last sampled tokens
            # and draft tokens. Also, get the logits indices to sample tokens from.
            logits_indices = combine_sampled_and_draft_tokens(
                self.input_buffers.input_ids,
                idx_mapping,
                self.req_states.last_sampled_tokens,
                query_start_loc,
                seq_lens,
                self.req_states.prefill_len.gpu,
                self.req_states.draft_tokens,
                cu_num_logits,
                total_num_logits,
                self.model_state.num_new_sampled_tokens_per_step,
            )

            # CPU upper bound on seq_lens (num_computed_tokens + num_scheduled_tokens).
            # Added by vLLM PR #40654 to avoid GPU->CPU sync for seq_lens.
            seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
            np.add(
                self.req_states.num_computed_tokens_np[idx_mapping_np],
                num_scheduled_tokens,
                out=seq_lens_cpu_upper_bound_np[:num_reqs],
            )
            seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)
            num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]

            max_seq_len_np = None
            if self.use_pp:
                # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
                max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

            prompt_lens = None
            if self.model_config.rswa_window is not None:
                # prompt_lens is only used in R-SWA case.
                prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

            input_batch = AscendInputBatch(
                req_ids=req_ids,
                num_reqs=num_reqs,
                num_reqs_after_padding=num_reqs_padded,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                expanded_idx_mapping=expanded_idx_mapping,
                expanded_local_pos=expanded_local_pos,
                num_scheduled_tokens=num_scheduled_tokens,
                num_tokens=num_tokens,
                num_tokens_after_padding=num_tokens_after_padding,
                num_draft_tokens=total_num_draft_tokens,
                num_draft_tokens_per_req=num_draft_tokens_per_req,
                query_start_loc=query_start_loc,
                query_start_loc_np=query_start_loc_np,
                seq_lens=seq_lens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
                is_prefilling_np=is_prefilling_np,
                has_prefill=batch_has_prefill,
                num_computed_tokens_np=num_computed_tokens_np,
                prefill_len_np=prefill_len_np,
                num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
                max_seq_len_np=max_seq_len_np,
                input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
                positions=self.input_buffers.positions[:num_tokens_after_padding],
                is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
                logits_indices=logits_indices,
                cu_num_logits=cu_num_logits,
                cu_num_logits_np=cu_num_logits_np,
                has_structured_output_reqs=scheduler_output.has_structured_output_requests,
                # TODO: only populated for R-SWA (not supported yet).
                prompt_lens=prompt_lens,
                # extra attributes for ascend npus.
                seq_lens_np=self.input_buffers.seq_lens_np,
                attn_state=attn_state,
            )

            input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

            # For mla/sfa, update cos/sin. Here is for execute_model.
            update_cos_sin(input_batch.positions)

            return input_batch

    def postprocess_sampled(
        self,
        idx_mapping,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc=None,
    ):
        """Override GPUModelRunner.postprocess_sampled for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to copy num_computed_tokens back to cpu here.
        """
        super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )

        # Skip D2H copy without MTP: num_computed_tokens_cpu is synced
        # from num_computed_tokens_np in _update_seq_lens_cpu instead.
        if self.speculator is not None:
            self._copy_num_computed_tokens_to_cpu()

    def _copy_num_computed_tokens_to_cpu(self):
        # npu attention backend still need to use seq_lens_cpu,
        # we need to copy num_computed_tokens back to cpu.
        default_stream = torch.cuda.current_stream()
        assert self.num_computed_tokens_stream is not None
        assert self.num_computed_tokens_cpu is not None
        with torch.npu.stream(self.num_computed_tokens_stream):
            self.num_computed_tokens_stream.wait_stream(default_stream)
            self.num_computed_tokens_cpu.copy_(
                self.req_states.num_computed_tokens.gpu,
                non_blocking=True,
            )
            self.num_computed_tokens_event.record()

    def _update_seq_lens_cpu(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[str],
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        # MTP needs D2H copy to get reverted num_computed_tokens after rejection.
        # Without MTP, num_computed_tokens_np is already correct from update_requests.
        if self.speculator is not None:
            self.num_computed_tokens_event.synchronize()
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]
        else:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.req_states.num_computed_tokens_np[req_index]

        # update seq_lens_cpu
        for i, req_id in enumerate(req_ids):  # type: ignore
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[req_index]
            self.input_buffers.seq_lens_cpu[i] = num_computed_tokens + num_scheduled_tokens[req_id]
        # Dummy FULL-graph / FIA pad rows must not keep the previous step's length.
        cpu_seq = self.input_buffers.seq_lens_cpu
        if len(req_ids) < len(cpu_seq):
            cpu_seq[len(req_ids) :] = 0

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        if (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        ):
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        if num_tokens_padded == num_reqs_padded * self.decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * self.decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        return query_start_loc_np, num_reqs_padded


@contextmanager
def graph_manager_wrapper(model_runner):
    """Context manager to override graph manager."""
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    if vllm_version_is("0.27.1"):

        def factory(
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
        ):
            return ModelAclGraphManager(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases=lora_capture_cases,
            )

    else:

        def factory(  # type: ignore[misc]
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
            varlen_decode: bool = False,
        ):
            return ModelAclGraphManager(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases=lora_capture_cases,
                varlen_decode=varlen_decode,  # type: ignore[call-arg]
            )

    try:
        vllm_model_runner.ModelCudaGraphManager = factory
        yield
    finally:
        vllm_model_runner.ModelCudaGraphManager = original_graph_manager
