# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# try:
#     from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
#
#
# import math
# from functools import partial
# import json
# import os
#
# from collections import namedtuple
#
# import torch
# import torch.nn as nn
#
# from mamba_ssm.models.config_mamba import MambaConfig
# from mamba_ssm.modules.mamba_simple import Mamba, Block
# from layers.mamba_simple import Mamba, Block
from layers.mamba_ssm.mamba_simple import Mamba
# from mamba_ssm.utils.generation import GenerationMixin
# from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from utils.masking import random_shuffle, unshuffle
from einops import rearrange, reduce

# import networkx as nx
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing, \
    solve_tsp_lin_kernighan


def create_block(
        d_model,
        model_name="Mamba",
        n_vars=None,
        shuffle_mode=0,
        mamba_mode=0,
        dropout=0.,
        use_casual_conv=True,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if model_name == "Mamba":
        from layers.mamba_ssm.mamba_simple import Block
        mixer_cls = partial(Mamba, layer_idx=layer_idx, dropout=dropout, use_casual_conv=use_casual_conv,
                            n_vars=n_vars, shuffle_mode=shuffle_mode,
                            **ssm_cfg, **factory_kwargs)
    elif model_name == "Mamba2":
        from layers.mamba_ssm.mamba2_simple import Mamba2TS as Block
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, dropout=dropout, use_casual_conv=use_casual_conv,
                            n_vars=n_vars, shuffle_mode=shuffle_mode,
                            **ssm_cfg, **factory_kwargs)
    elif model_name == "MambaTS":
        from layers.mamba_ssm.mamba_simple import MambaTS
        from layers.mamba_ssm.mamba_simple import BlockTS as Block
        mixer_cls = partial(MambaTS, layer_idx=layer_idx, mamba_mode=mamba_mode, n_vars=n_vars,
                            **ssm_cfg, **factory_kwargs)
    elif model_name == "MambaQK":
        from layers.mamba_ssm.mamba_simple import MambaQK
        from layers.mamba_ssm.mamba_simple import BlockQK as Block
        mixer_cls = partial(MambaQK, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        raise NotImplementedError
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            vocab_size: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MixerTSModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            n_vars=None,
            model_name='Mamba',
            arrange_mode=0,
            mamba_mode=0,
            shuffle_mode=0,
            dropout=0.,
            ssm_cfg: int = None,
            norm_epsilon: float = 1e-5,
            use_casual_conv: bool = True,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.n_vars = n_vars

        # shuffle_mode=0, no shuffle
        # shuffle_mode=1, input-level shuffle
        # shuffle_mode=2, layer-level shuffle
        # shuffle_mode=3, layer-level channel-wise shuffle
        self.arrange_mode = arrange_mode
        self.mamba_mode = mamba_mode
        self.shuffle_mode = shuffle_mode
        print(f"mamba_mode: {mamba_mode}, shuffle_mode: {shuffle_mode}")
        if self.shuffle_mode > 0:
            assert n_vars is not None, f"When shuffle_mode ({self.shuffle_mode}) > 0, n_vars should be passed in."
            self.d_model = d_model

        if self.shuffle_mode == 8:
            self.target = torch.randperm(self.n_vars).cuda()
        if self.shuffle_mode == 5:
            self.ids_shuffle = None
        self.ids_shuffle = None
        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # topological sorting
        self.adjacency_matrix = nn.Parameter(torch.zeros(n_vars, n_vars), requires_grad=False)
        self.count_matrix = nn.Parameter(torch.zeros(n_vars, n_vars), requires_grad=False)
        self.ending_points = nn.Parameter(torch.zeros(n_vars), requires_grad=False)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    model_name=model_name,
                    n_vars=self.n_vars,
                    shuffle_mode=self.shuffle_mode,
                    dropout=dropout,
                    use_casual_conv=use_casual_conv,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x, inference_params=None):
        # if self.shuffle_mode == 4 and not self.training:
        #     self.shuffle_mode = 44
        # if self.shuffif self.args.agg_mode:fle_mode = 4
        # if self.shuffle_mode == 44 and self.training:
        #     self.shuffle_mode = 4
        if self.shuffle_mode == 5 and not self.training:
            self.shuffle_mode = 55
            if self.ids_shuffle is not None:
                self.ids_shuffle = None
        elif self.shuffle_mode == 55 and self.training:
            self.shuffle_mode = 5
        # input-level shuffle
        if self.shuffle_mode == 1:
            x, ids_restore = random_shuffle(x, mask_ratio=0)
        elif self.shuffle_mode in [5, 55, 51, 52, 53, 54, 56] and self.arrange_mode == 0:
            if self.training:
                x = rearrange(x, 'b (c l) d -> b c (l d)', c=self.n_vars)
                x, ids_shuffle, ids_restore = random_shuffle(x, mask_ratio=0, return_ids_shuffle=True)
                # record the starting point
                ids_shuffle_ = torch.cat([ids_shuffle[:, [0]], ids_shuffle], dim=-1)
                self.transition_tuple = torch.stack([ids_shuffle_[:, :-1], ids_shuffle_[:, 1:]], dim=-1)
                x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
                self.ids_shuffle = None
            else:
                if self.arrange_mode == 0:
                    x = rearrange(x, 'b (c l) d -> b c (l d)', c=self.n_vars)
                else:
                    x = rearrange(x, 'b (l c) d -> b c (l d)', c=self.n_vars)

                if self.shuffle_mode == 51:
                    mode = 'greedy'
                elif self.shuffle_mode == 55:
                    mode = 'sa'
                elif self.shuffle_mode == 56:
                    mode = 'sans'
                elif self.shuffle_mode == 52:
                    mode = 'ls'
                elif self.shuffle_mode == 53:
                    mode = 'lk'

                if self.ids_shuffle is None:
                    if mode == 'greedy':
                        # 起始节点
                        adjacency_matrix = self.adjacency_matrix.clone()
                        start_node = torch.argmin(torch.diag(adjacency_matrix)).item()

                        for j in range(adjacency_matrix.size(0)):
                            adjacency_matrix[j, j] = float("inf")
                        current_node = start_node
                        visited_nodes = [current_node]
                        # 遍历所有节点
                        while len(visited_nodes) < adjacency_matrix.size(0):
                            # 将当前节点标记为已访问
                            adjacency_matrix[:, current_node] = float('inf')

                            # 找到离当前节点最近的节点
                            next_node = torch.argmin(adjacency_matrix[current_node]).item()

                            # 更新当前节点为下一个节点
                            current_node = next_node
                            visited_nodes.append(current_node)

                        # print("遍历顺序：", visited_nodes)
                        ids_shuffle = torch.tensor(visited_nodes, device=x.device)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                    elif mode == 'topo_sort':
                        # 起始节点
                        adjacency_matrix = self.adjacency_matrix.clone()
                        start_node = torch.argmin(torch.diag(adjacency_matrix)).item()

                        G = self.construct_graph(adjacency_matrix, do_ReLU=True)
                        # print(G)
                        traversal_path = list(nx.topological_sort(G))
                        # print(len(traversal_path), self.n_vars, print(traversal_path))

                        # print("遍历顺序：", visited_nodes)
                        ids_shuffle = torch.tensor(traversal_path, device=x.device)

                        # ids_shuffle = self.get_short_path(adjacency_matrix)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                        # print(self.ids_shuffle.shape)
                    elif mode == 'tsp':
                        # 起始节点
                        adjacency_matrix = self.adjacency_matrix.clone()
                        adjacency_matrix[adjacency_matrix > adjacency_matrix.mean()] = 0
                        print(adjacency_matrix)

                        G = self.construct_graph(adjacency_matrix, do_ReLU=False)
                        # print(G)
                        shortest_path = nx.approximation.traveling_salesman_problem(G, cycle=False, weight='weight').cuda()
                        # print(len(traversal_path), self.n_vars, print(traversal_path))

                        # print("遍历顺序：", visited_nodes)
                        ids_shuffle = torch.tensor(shortest_path, device=x.device)

                        # ids_shuffle = self.get_short_path(adjacency_matrix)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                        # print(self.ids_shuffle.shape)
                    elif mode == 'sa':
                        # 起始节点
                        adjacency_matrix = self.adjacency_matrix.clone() / self.count_matrix

                        # # 起始节点
                        start_index = torch.argmin(torch.diag(adjacency_matrix)).item()
                        for j in range(adjacency_matrix.size(0)):
                            adjacency_matrix[j, j] = 0

                        distance_matrix = adjacency_matrix.cpu().numpy()
                        distance_matrix[:, start_index] = 0


                        # shortest_path, distance = solve_tsp_dynamic_programming(distance_matrix)
                        # shortest_path, distance = solve_tsp_branch_and_bound(distance_matrix)
                        # shortest_path, distance = solve_tsp_brute_force(distance_matrix)
                        shortest_path, distance = solve_tsp_simulated_annealing(distance_matrix, x0=list(range(self.n_vars)))
                        # shortest_path, distance = solve_tsp_local_search(distance_matrix)
                        # shortest_path, distance = solve_tsp_record_to_record(distance_matrix)
                        # shortest_path, distance = solve_tsp_lin_kernighan(distance_matrix)
                        print(shortest_path, distance)

                        # print("遍历顺序：", visited_nodes)
                        ids_shuffle = torch.tensor(shortest_path, device=x.device)

                        # ids_shuffle = self.get_short_path(adjacency_matrix)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                        # print(self.ids_shuffle.shape, x.shape)
                    elif mode == 'sans':
                        # 起始节点
                        adjacency_matrix = self.adjacency_matrix.clone() / self.count_matrix

                        # # 起始节点
                        # start_index = torch.argmin(torch.diag(adjacency_matrix)).item()
                        for j in range(adjacency_matrix.size(0)):
                            adjacency_matrix[j, j] = 0

                        distance_matrix = adjacency_matrix.cpu().numpy()
                        # distance_matrix[:, start_index] = 0

                        # shortest_path, distance = solve_tsp_dynamic_programming(distance_matrix)
                        # shortest_path, distance = solve_tsp_branch_and_bound(distance_matrix)
                        # shortest_path, distance = solve_tsp_brute_force(distance_matrix)
                        shortest_path, distance = solve_tsp_simulated_annealing(distance_matrix, x0=list(range(self.n_vars)))
                        # shortest_path, distance = solve_tsp_local_search(distance_matrix)
                        # shortest_path, distance = solve_tsp_record_to_record(distance_matrix)
                        # shortest_path, distance = solve_tsp_lin_kernighan(distance_matrix)
                        print(shortest_path, distance)

                        # print("遍历顺序：", visited_nodes)
                        ids_shuffle = torch.tensor(shortest_path, device=x.device)

                        # ids_shuffle = self.get_short_path(adjacency_matrix)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                        # print(self.ids_shuffle.shape, x.shape)
                    elif mode == 'lk':
                        # 起始节点
                        adjacency_matrix = self.adjacency_matrix.clone() / self.count_matrix

                        # # 起始节点
                        start_index = torch.argmin(torch.diag(adjacency_matrix)).item()
                        for j in range(adjacency_matrix.size(0)):
                            adjacency_matrix[j, j] = 0

                        distance_matrix = adjacency_matrix.cpu().numpy()
                        distance_matrix[:, start_index] = 0

                        # shortest_path, distance = solve_tsp_dynamic_programming(distance_matrix)
                        # shortest_path, distance = solve_tsp_branch_and_bound(distance_matrix)
                        # shortest_path, distance = solve_tsp_brute_force(distance_matrix)
                        # shortest_path, distance = solve_tsp_simulated_annealing(distance_matrix, x0=list(range(self.n_vars)))
                        # shortest_path, distance = solve_tsp_local_search(distance_matrix)
                        # shortest_path, distance = solve_tsp_record_to_record(distance_matrix)
                        shortest_path, distance = solve_tsp_lin_kernighan(distance_matrix, x0=list(range(self.n_vars)))
                        print(shortest_path, distance)

                        # print("遍历顺序：", visited_nodes)
                        ids_shuffle = torch.tensor(shortest_path, device=x.device)

                        # ids_shuffle = self.get_short_path(adjacency_matrix)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                        # print(self.ids_shuffle.shape, x.shape)
                    elif mode == 'ls':
                        # 起始节点
                        adjacency_matrix = self.adjacency_matrix.clone() / self.count_matrix

                        # # 起始节点
                        start_index = torch.argmin(torch.diag(adjacency_matrix)).item()
                        for j in range(adjacency_matrix.size(0)):
                            adjacency_matrix[j, j] = 0

                        distance_matrix = adjacency_matrix.cpu().numpy()
                        distance_matrix[:, start_index] = 0


                        # shortest_path, distance = solve_tsp_dynamic_programming(distance_matrix)
                        # shortest_path, distance = solve_tsp_branch_and_bound(distance_matrix)
                        # shortest_path, distance = solve_tsp_brute_force(distance_matrix)
                        # shortest_path, distance = solve_tsp_simulated_annealing(distance_matrix, x0=list(range(self.n_vars)))
                        shortest_path, distance = solve_tsp_local_search(distance_matrix)
                        # shortest_path, distance = solve_tsp_record_to_record(distance_matrix)
                        # shortest_path, distance = solve_tsp_lin_kernighan(distance_matrix)
                        print(shortest_path, distance)

                        # print("遍历顺序：", visited_nodes)
                        ids_shuffle = torch.tensor(shortest_path, device=x.device)

                        # ids_shuffle = self.get_short_path(adjacency_matrix)

                        self.ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
                        # print(self.ids_shuffle.shape, x.shape)
                # ids_restore = torch.argsort(self.ids_shuffle, dim=1, descending=False)
                x, ids_restore = random_shuffle(x, ids_shuffle=self.ids_shuffle)
                if self.arrange_mode == 0:
                    x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
                else:
                    x = rearrange(x, 'b c (l d) -> b (l c) d', d=self.d_model)

        elif self.shuffle_mode == 4 and self.arrange_mode == 0:
            x = rearrange(x, 'b (c l) d -> b c (l d)', c=self.n_vars)
            x, ids_restore = random_shuffle(x, mask_ratio=0)
            x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
        elif self.shuffle_mode == 4 and self.arrange_mode == 1:
            x = rearrange(x, 'b (l c) d -> b c (l d)', c=self.n_vars)
            x, ids_restore = random_shuffle(x, mask_ratio=0)
            x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
        elif self.shuffle_mode in [44, 441, 442, 443, 41, 43, 8]:
            # 4 for VAST, 1234 for \DeltaABC, _123 for _max 1mean 2max+mean 3max+rev
            if self.arrange_mode == 0:
                x = rearrange(x, 'b (c l) d -> b c (l d)', c=self.n_vars)
            else:
                x = rearrange(x, 'b (l c) d -> b c (l d)', c=self.n_vars)

            ci_dt, ci_B, ci_C = None, None, None
            for i in range(len(self.layers)):
                if ci_dt is None:
                    ci_dt = self.layers[i].mixer.ci_dt
                    ci_B = self.layers[i].mixer.ci_B
                    ci_C = self.layers[i].mixer.ci_C
                else:
                    ci_dt += self.layers[i].mixer.ci_dt
                    ci_B += self.layers[i].mixer.ci_B
                    ci_C += self.layers[i].mixer.ci_C

            if self.shuffle_mode == 44:
                target = ci_C[1]
            elif self.shuffle_mode == 441:
                target = ci_C[0]
            elif self.shuffle_mode == 442:
                target = ci_C[1] + ci_C[0]
            elif self.shuffle_mode == 443:
                target = - ci_C[1]
            elif self.shuffle_mode == 41:
                target = ci_dt[1]
            elif self.shuffle_mode == 43:
                target = ci_B[1]
            elif self.shuffle_mode == 8:
                target = self.target
                # print(target.shape)
                # print(target)
                # print(target[0])

            ids_shuffle = torch.argsort(target, dim=-1)  # ascend: small is keep, large is remove
            # print(ids_shuffle)
            ids_shuffle = ids_shuffle.repeat(x.shape[0], 1)
            ids_restore = torch.argsort(ids_shuffle, dim=1, descending=False)
            x, _ = random_shuffle(x, ids_shuffle=ids_shuffle)
            if self.arrange_mode == 0:
                x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
            else:
                x = rearrange(x, 'b c (l d) -> b (l c) d', d=self.d_model)
        else:
            ids_restore = None
        hidden_states = x  # BxLxD
        residual = None
        residual_1 = None
        residual_2 = None
        ret_list = []
        if self.mamba_mode == 4:  # PatchTST-like
            hidden_states = rearrange(hidden_states, 'b (c l) d -> (b c) l d', c=self.n_vars)
        for layer in self.layers:
            # layer-level shuffle
            if self.shuffle_mode == 2:
                hidden_states, ids_restore = random_shuffle(hidden_states, mask_ratio=0)
            elif self.shuffle_mode == 3:
                hidden_states = rearrange(hidden_states, 'b (c l) d -> b c (l d)', c=self.n_vars)
                hidden_states, ids_restore = random_shuffle(hidden_states, mask_ratio=0)
                hidden_states = rearrange(hidden_states, 'b c (l d) -> b (c l) d', d=self.d_model)

            if self.mamba_mode == 0:  # global modeling
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, ids_restore=ids_restore
                )
            elif self.mamba_mode == 4:  # PatchTST-like
                # hidden_states = rearrange(hidden_states, 'b (c l) d -> (b c) l d', c=self.n_vars)
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
                # hidden_states = rearrange(hidden_states, '(b c) l d -> b (c l) d', c=self.n_vars)
            elif self.mamba_mode == 1:  # concurrent
                b = hidden_states.shape[0]
                hidden_states_1 = rearrange(hidden_states, 'b (c l) d -> (b c) l d', c=self.n_vars)
                # layer-level shuffle
                # if self.shuffle_mode == 5:
                #     hidden_states_1, ids_restore = random_shuffle(hidden_states_1, mask_ratio=0)
                # elif self.shuffle_mode == 6:
                #     hidden_states_1, ids_restore = random_shuffle(hidden_states_1, ids_shuffle=ids_shuffle_1)
                hidden_states_1, residual_1 = layer(
                    hidden_states_1, residual_1, inference_params=inference_params
                )
                # layer-level un-shuffle
                # if self.shuffle_mode in [5, 6]:
                #     hidden_states_1 = unshuffle(hidden_states_1, ids_restore)
                hidden_states_1 = rearrange(hidden_states_1, '(b c) l d -> b (c l) d', b=b)

                hidden_states_2 = rearrange(hidden_states, 'b (c l) d -> (b l) c d', c=self.n_vars)
                # layer-level shuffle
                # if self.shuffle_mode == 5:
                #     hidden_states_2, ids_restore = random_shuffle(hidden_states_2, mask_ratio=0)
                # if self.shuffle_mode == 6:
                #     hidden_states_2, ids_restore = random_shuffle(hidden_states_2, ids_shuffle=ids_shuffle_2)
                hidden_states_2, residual_2 = layer(
                    hidden_states_2, residual_2, inference_params=inference_params
                )
                # layer-level un-shuffle
                # if self.shuffle_mode in [5, 6]:
                #     hidden_states_2 = unshuffle(hidden_states_2, ids_restore)
                hidden_states_2 = rearrange(hidden_states_2, '(b l) c d -> b (c l) d', b=b)

                hidden_states = hidden_states_1 + hidden_states_2
            elif self.mamba_mode == 2:  # time along -> channel along
                b = hidden_states.shape[0]
                hidden_states_1 = rearrange(hidden_states, 'b (c l) d -> (b c) l d', c=self.n_vars)
                hidden_states_1, residual_1 = layer(
                    hidden_states_1, residual_1, inference_params=inference_params
                )
                hidden_states_2 = rearrange(hidden_states_1, '(b c) l d -> (b l) c d', c=self.n_vars)
                hidden_states_2, residual_2 = layer(
                    hidden_states_2, residual_2, inference_params=inference_params
                )
                hidden_states_2 = rearrange(hidden_states_2, '(b l) c d -> b (c l) d', b=b)

                hidden_states = hidden_states_2
            elif self.mamba_mode == 3:  # channel along -> time along
                b = hidden_states.shape[0]
                hidden_states_2 = rearrange(hidden_states, 'b (c l) d -> (b l) c d', c=self.n_vars)
                hidden_states_2, residual_2 = layer(
                    hidden_states_2, residual_2, inference_params=inference_params
                )
                hidden_states_1 = rearrange(hidden_states_2, '(b l) c d -> (b c) l d', b=b)
                hidden_states_1, residual_1 = layer(
                    hidden_states_1, residual_1, inference_params=inference_params
                )
                hidden_states_1 = rearrange(hidden_states_1, '(b c) l d -> b (c l) d', b=b)

                hidden_states = hidden_states_1

            # layer-level un-shuffle
            if self.shuffle_mode == 2:
                hidden_states = unshuffle(hidden_states, ids_restore)
            elif self.shuffle_mode == 3:
                hidden_states = rearrange(hidden_states, 'b (c l) d -> b c (l d)', c=self.n_vars)
                hidden_states = unshuffle(hidden_states, ids_restore)
                hidden_states = rearrange(hidden_states, 'b c (l d) -> b (c l) d', d=self.d_model)
            # ret_list.append(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # ret_list[-1] = hidden_states
        if self.mamba_mode == 4:
            hidden_states = rearrange(hidden_states, '(b c) l d -> b (c l) d', c=self.n_vars)
        # input-level un-shuffle
        if self.shuffle_mode == 1:
            hidden_states = unshuffle(hidden_states, ids_restore)
        elif self.shuffle_mode not in [0, 1] and self.arrange_mode == 0:
            hidden_states = rearrange(hidden_states, 'b (c l) d -> b c (l d)', c=self.n_vars)
            hidden_states = unshuffle(hidden_states, ids_restore)
            hidden_states = rearrange(hidden_states, 'b c (l d) -> b (c l) d', d=self.d_model)
        elif self.shuffle_mode not in [0, 1] and self.arrange_mode == 1:
            hidden_states = rearrange(hidden_states, 'b (l c) d -> b c (l d)', c=self.n_vars)
            hidden_states = unshuffle(hidden_states, ids_restore)
            hidden_states = rearrange(hidden_states, 'b c (l d) -> b (l c) d', d=self.d_model)
        # print(self.shuffle_mode)

        return hidden_states, ret_list

    def batch_update_state(self, cost_tensor):
        cost_tensor = cost_tensor.detach()
        cost_tensor = reduce(cost_tensor, 'b t c -> b', 'mean')
        cost_tensor = cost_tensor - cost_tensor.mean()
        count_tensor = torch.ones_like(cost_tensor)
        # cost_tensor[cost_tensor > 0] = 0

        B, C = cost_tensor.size(0), self.adjacency_matrix.size(0)

        # 计算每个坐标在state_tensor中的索引
        indices = self.transition_tuple[:, :, 0] * C + self.transition_tuple[:, :, 1]

        # 将cost_tensor按照indices散布到state_tensor中
        self.adjacency_matrix.view(-1).scatter_add_(0, indices.view(-1), cost_tensor.repeat(indices.shape[-1]))
        self.count_matrix.view(-1).scatter_add_(0, indices.view(-1), count_tensor.repeat(indices.shape[-1]))

        # update the ending points
        self.ending_points.scatter_add_(0, self.transition_tuple[:, -1, 1], cost_tensor)

    def update_best_state(self):
        self.best_adjacency_matrix.data = self.adjacency_matrix.data.clone()

    def construct_graph(self, adjacency_matrix, do_ReLU=True):

        # 创建一个有向图
        G = nx.DiGraph()

        # 假设有一个CxC的邻接矩阵A，表示从节点i到节点j的代价
        # 这里假设邻接矩阵A是一个numpy数组
        # A = np.array([[0, 1, 3],
        #               [2, 0, 4],
        #               [5, 6, 0]])
        # print(adjacency_matrix)
        if do_ReLU:
            adjacency_matrix = - torch.relu(- adjacency_matrix)
            # print(adjacency_matrix)
        else:
            adjacency_matrix = adjacency_matrix

        adjacency_matrix[adjacency_matrix < 0] = adjacency_matrix[adjacency_matrix < 0] + (abs(adjacency_matrix.min()) + 1e-7)

        A = adjacency_matrix.cpu().numpy()
        # print(A)
        # 添加节点和边到有向图
        # print(f"constructing the graph...")
        num_nodes = len(A)
        for i in range(num_nodes):
            G.add_node(i)
            for j in range(num_nodes):
                # if A[i][j] != 0:
                if i != j and A[i][j] != 0:
                    G.add_edge(i, j, weight=A[i][j])

        return G

    def get_short_path(self, adjacency_matrix, do_ReLU=True):
        import networkx as nx

        # 创建一个有向图
        G = nx.DiGraph()

        # 假设有一个CxC的邻接矩阵A，表示从节点i到节点j的代价
        # 这里假设邻接矩阵A是一个numpy数组
        # A = np.array([[0, 1, 3],
        #               [2, 0, 4],
        #               [5, 6, 0]])
        # print(adjacency_matrix)
        if do_ReLU:
            adjacency_matrix = - torch.relu(- adjacency_matrix)
            print(adjacency_matrix)
        else:
            adjacency_matrix = adjacency_matrix

        adjacency_matrix[adjacency_matrix < 0] = adjacency_matrix[adjacency_matrix < 0] + (abs(adjacency_matrix.min()) + 1e-7)

        A = adjacency_matrix.cpu().numpy()
        # print(A)
        # 添加节点和边到有向图
        print(f"constructing the graph...")
        num_nodes = len(A)
        for i in range(num_nodes):
            G.add_node(i)
            for j in range(num_nodes):
                # if A[i][j] != 0:
                if i != j and A[i][j] != 0:
                    G.add_edge(i, j, weight=A[i][j])

        print(f"finding the shortest path...")
        # 找到遍历所有节点的最短路径
        shortest_path = nx.approximation.traveling_salesman_problem(G, cycle=False, weight='weight')

        # print("最短路径：", shortest_path)
        shortest_path = torch.from_numpy(shortest_path)
        return shortest_path


class MixerTSModelV1(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            n_vars=None,
            model_name='Mamba',
            mamba_mode=0,
            shuffle_mode=0,
            ssm_cfg: int = None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.n_vars = n_vars

        # shuffle_mode=0, no shuffle
        # shuffle_mode=1, input-level shuffle
        # shuffle_mode=2, layer-level shuffle
        # shuffle_mode=3, layer-level channel-wise shuffle
        self.mamba_mode = mamba_mode
        self.shuffle_mode = shuffle_mode
        print(f"mamba_mode: {mamba_mode}, shuffle_mode: {shuffle_mode}")
        if self.shuffle_mode > 0:
            assert n_vars is not None, f"When shuffle_mode ({self.shuffle_mode}) > 0, n_vars should be passed in."
            self.d_model = d_model

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    model_name=model_name,
                    n_vars=self.n_vars,
                    mamba_mode=self.mamba_mode,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x, inference_params=None):
        # input-level shuffle
        # if self.shuffle_mode == 1:
        #     x, ids_restore = random_shuffle(x, mask_ratio=0)
        # elif self.shuffle_mode == 4:
        #     x = rearrange(x, 'b (c l) d -> b c (l d)', c=self.n_vars)
        #     x, ids_restore = random_shuffle(x, mask_ratio=0)
        #     x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
        x = rearrange(x, 'b (c l) d -> (b c) l d', c=self.n_vars)
        hidden_states = x  # BxLxD
        residual = None
        ret_list = []
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            # ret_list.append(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # ret_list[-1] = hidden_states
        # input-level un-shuffle
        # if self.shuffle_mode == 1:
        #     hidden_states = unshuffle(hidden_states, ids_restore)
        # elif self.shuffle_mode == 4:
        #     hidden_states = rearrange(hidden_states, 'b (c l) d -> b c (l d)', c=self.n_vars)
        #     hidden_states = unshuffle(hidden_states, ids_restore)
        #     hidden_states = rearrange(hidden_states, 'b c (l d) -> b (c l) d', d=self.d_model)
        hidden_states = rearrange(hidden_states, '(b c) l d -> b (c l) d', c=self.n_vars)
        return hidden_states, ret_list


class MixerTSDecoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            n_vars=None,
            shuffle_mode=0,
            ssm_cfg: int = None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.n_vars = n_vars

        # shuffle_mode=0, no shuffle
        # shuffle_mode=1, input-level shuffle
        # shuffle_mode=2, layer-level shuffle
        # shuffle_mode=3, layer-level channel-wise shuffle
        self.shuffle_mode = shuffle_mode
        if self.shuffle_mode > 0:
            assert n_vars is not None, f"When shuffle_mode ({self.shuffle_mode}) > 0, n_vars should be passed in."
            self.d_model = d_model

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    model_name="MambaQK",
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, q, kv, inference_params=None):
        if not isinstance(kv, (list, tuple)):
            kv_list = [kv for _ in range(len(self.layers))]
        else:
            kv_list = kv
        # input-level shuffle
        # if self.shuffle_mode == 1:
        #     x, ids_restore = random_shuffle(x, mask_ratio=0)
        # elif self.shuffle_mode == 4:
        #     x = rearrange(x, 'b (c l) d -> b c (l d)', c=self.n_vars)
        #     x, ids_restore = random_shuffle(x, mask_ratio=0)
        #     x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)
        query_states = q  # BxLxD
        # hidden_states = kv  # BxLxD
        residual = None
        for kv, layer in zip(kv_list, self.layers):
            # layer-level shuffle
            # if self.shuffle_mode == 2:
            #     x, ids_restore = random_shuffle(x, mask_ratio=0)
            # elif self.shuffle_mode == 3:
            #     x = rearrange(x, 'b (c l) d -> b c (l d)', c=self.n_vars)
            #     x, ids_restore = random_shuffle(x, mask_ratio=0)
            #     x = rearrange(x, 'b c (l d) -> b (c l) d', d=self.d_model)

            query_states, residual = layer(
                query_states, kv, residual, inference_params=inference_params
            )

            # hidden_states = hidden_states + hidden_states2
            # residual = residual + residual2
            # layer-level un-shuffle
            # if self.shuffle_mode == 2:
            #     hidden_states = unshuffle(hidden_states, ids_restore)
            # elif self.shuffle_mode == 3:
            #     hidden_states = rearrange(hidden_states, 'b (c l) d -> b c (l d)', c=self.n_vars)
            #     hidden_states = unshuffle(hidden_states, ids_restore)
            #     hidden_states = rearrange(hidden_states, 'b c (l d) -> b (c l) d', d=self.d_model)
        if not self.fused_add_norm:
            residual = (query_states + residual) if residual is not None else query_states
            query_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            query_states = fused_add_norm_fn(
                query_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # input-level un-shuffle
        # if self.shuffle_mode == 1:
        #     hidden_states = unshuffle(hidden_states, ids_restore)
        # elif self.shuffle_mode == 4:
        #     hidden_states = rearrange(hidden_states, 'b (c l) d -> b c (l d)', c=self.n_vars)
        #     hidden_states = unshuffle(hidden_states, ids_restore)
        #     hidden_states = rearrange(hidden_states, 'b c (l d) -> b (c l) d', d=self.d_model)
        return query_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)
