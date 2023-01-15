import torch

from lora import LoRALinear


def add_lora_to_gpt_j(
    model, adapter_rank, device="cuda" if torch.cuda.is_available() else "cpu"
):
    for i in range(len(model.transformer.h)):
        lora_layer = LoRALinear(
            model.transformer.h[i].attn.k_proj.in_features,
            model.transformer.h[i].attn.k_proj.out_features,
            adapter_rank,
        ).to(device)
        lora_layer.main.weight.data = model.transformer.h[i].attn.k_proj.weight.data
        lora_layer.main.bias = model.transformer.h[i].attn.k_proj.bias
        model.transformer.h[i].attn.k_proj = lora_layer

        lora_layer = LoRALinear(
            model.transformer.h[i].attn.v_proj.in_features,
            model.transformer.h[i].attn.v_proj.out_features,
            adapter_rank,
        ).to(device)
        lora_layer.main.weight.data = model.transformer.h[i].attn.v_proj.weight.data
        lora_layer.main.bias = model.transformer.h[i].attn.v_proj.bias
        model.transformer.h[i].attn.v_proj = lora_layer

        lora_layer = LoRALinear(
            model.transformer.h[i].attn.q_proj.in_features,
            model.transformer.h[i].attn.q_proj.out_features,
            adapter_rank,
        ).to(device)
        lora_layer.main.weight.data = model.transformer.h[i].attn.q_proj.weight.data
        lora_layer.main.bias = model.transformer.h[i].attn.q_proj.bias
        model.transformer.h[i].attn.q_proj = lora_layer
