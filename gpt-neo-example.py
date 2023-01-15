from lora.gpt_neo import add_lora_to_gpt_neo
from lora import save_lora_layers, load_lora_layers
from transformers import AutoModelForCausalLM, AutoTokenizer

# load a gpt neo model
model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt_neo")
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt_neo")

# add low rank adapters
add_lora_to_gpt_neo(model, 16)
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# train lora layers here however you want

# save the loras
save_lora_layers(model, "gpt-neo-finetuned-lora-layers.lora")

# test load the loras back in
model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt_neo")
add_lora_to_gpt_neo(model, 16)
load_lora_layers(model, "gpt-neo-finetuned-lora-layers.lora")
