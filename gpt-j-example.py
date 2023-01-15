from transformers import AutoModelForCausalLM, AutoTokenizer

from lora import load_lora_layers, save_lora_layers
from lora.gpt_j import add_lora_to_gpt_j

# load a gpt neo model
model = AutoModelForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random")
tokenizer = AutoTokenizer.from_pretrained("anton-l/gpt-j-tiny-random")

# add low rank adapters
add_lora_to_gpt_j(model, 16, device=model.device)
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# train lora layers here however you want

# save the loras
save_lora_layers(model, "gpt-j-finetuned-lora-layers.lora")

# test load the loras back in
model = AutoModelForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random")
add_lora_to_gpt_j(model, 16, device=model.device)
load_lora_layers(model, "gpt-j-finetuned-lora-layers.lora")
