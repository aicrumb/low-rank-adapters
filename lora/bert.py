import torch
from .utils import LoRALinear

# I don't really know which one is what we want? There's attn outputs, outputs, and the qkv projections
# I just said screw it and added them all
# Maybe giving it a "key" would be better? (add_lora(model, key="output.dense") or whatever)

# I found in one repo for stable diffusion + lora this is what's tuned
# so I'll leave this as the base name add_lora_to_bert because I'm assuming it's the standard
def add_lora_to_bert(model, adapter_rank):
    for i in range(len(model.encoder.layer)):
        # replace query
        lora_layer = LoRALinear(
            model.encoder.layer[i].attention.self.query.in_features, 
            model.encoder.layer[i].attention.self.query.out_features, 
            16
        )
        lora_layer.main.weight.data = model.encoder.layer[i].attention.self.query.weight.data
        if model.encoder.layer[i].attention.self.query.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].attention.self.query.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].attention.self.query = lora_layer

        # replace key
        lora_layer = LoRALinear(
            model.encoder.layer[i].attention.self.key.in_features, 
            model.encoder.layer[i].attention.self.key.out_features, 
            16
        )
        lora_layer.main.weight.data = model.encoder.layer[i].attention.self.key.weight.data
        if model.encoder.layer[i].attention.self.key.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].attention.self.key.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].attention.self.key = lora_layer

        # replace value
        lora_layer = LoRALinear(
            model.encoder.layer[i].attention.self.value.in_features, 
            model.encoder.layer[i].attention.self.value.out_features, 
            16
        )
        lora_layer.main.weight.data = model.encoder.layer[i].attention.self.value.weight.data
        if model.encoder.layer[i].attention.self.value.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].attention.self.value.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].attention.self.value = lora_layer

# I don't know if these are useful or not but I added them just in case, I don't have many BERT finetuning projects so I can't exactly test them

def add_lora_to_bert_output_layers(model, adapter_rank):
    for i in range(len(model.encoder.layer)):
        lora_layer = LoRALinear(
            model.encoder.layer[i].output.dense.in_features, 
            model.encoder.layer[i].output.dense.out_features, 
            adapter_rank
        )

        lora_layer.main.weight.data = model.encoder.layer[i].output.dense.weight.data
        if model.encoder.layer[i].output.dense.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].output.dense.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].output.dense = lora_layer

def add_lora_to_bert_attention_output_layers(model, adapter_rank):
    for i in range(len(model.encoder.layer)):
        lora_layer = LoRALinear(
            model.encoder.layer[i].attention.output.dense.in_features, 
            model.encoder.layer[i].attention.output.dense.out_features, 
            adapter_rank
        )

        lora_layer.main.weight.data = model.encoder.layer[i].attention.output.dense.weight.data
        if model.encoder.layer[i].attention.output.dense.bias != None:
            lora_layer.main.bias = model.encoder.layer[i].attention.output.dense.bias
        else:
            lora_layer.main.bias = None
        model.encoder.layer[i].attention.output.dense = lora_layer
