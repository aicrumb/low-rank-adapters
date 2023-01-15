import torch

from .utils import LoRALinear


def add_lora_to_bert(
    model, adapter_rank, device="cuda" if torch.cuda.is_available() else "cpu"
):
    for i in range(len(model.encoder.layer)):
        # replace query
        lora_layer = LoRALinear(
            model.encoder.layer[i].attention.self.query.in_features,
            model.encoder.layer[i].attention.self.query.out_features,
            adapter_rank,
        ).to(device)
        lora_layer.main.weight.data = model.encoder.layer[
            i
        ].attention.self.query.weight.data
        if model.encoder.layer[i].attention.self.query.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].attention.self.query.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].attention.self.query = lora_layer

        # replace key
        lora_layer = LoRALinear(
            model.encoder.layer[i].attention.self.key.in_features,
            model.encoder.layer[i].attention.self.key.out_features,
            adapter_rank,
        ).to(device)
        lora_layer.main.weight.data = model.encoder.layer[
            i
        ].attention.self.key.weight.data
        if model.encoder.layer[i].attention.self.key.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].attention.self.key.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].attention.self.key = lora_layer

        # replace value
        lora_layer = LoRALinear(
            model.encoder.layer[i].attention.self.value.in_features,
            model.encoder.layer[i].attention.self.value.out_features,
            adapter_rank,
        ).to(device)
        lora_layer.main.weight.data = model.encoder.layer[
            i
        ].attention.self.value.weight.data
        if model.encoder.layer[i].attention.self.value.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].attention.self.value.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].attention.self.value = lora_layer
