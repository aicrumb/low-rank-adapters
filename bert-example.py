import torch
from transformers import AutoModel, AutoTokenizer

from lora import load_lora_layers, save_lora_layers
from lora.bert import add_lora_to_bert

# load the base bert model
model_id = "bert-base-uncased"
model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# add low rank adapters and set requires_grad for each param
add_lora_to_bert(model, adapter_rank=16)
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# train lora layers here however you want

# save the loras
save_lora_layers(model, "bert_finetuned_lora_layers.lora")

# re-load bert and loras
model_id = "bert-base-uncased"
model = AutoModel.from_pretrained(model_id)
add_lora_to_bert(model, 16)
load_lora_layers(model, "bert_finetuned_lora_layers.lora")
