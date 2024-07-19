# Joint Compression Methods for PEFT

[Rickard Brüel Gabrielsson](http://bruel.org/), Jiacheng Zhu, Onkar Bhardwaj, Leshem Choshen, Kristjan Greenewald, Mikhail Yurochkin and Justin Solomon
[[arXiv](https://www.arxiv.org/abs/2407.00066)]

### Citation

```
@misc{brüelgabrielsson2024compressserveservingthousands,
      title={Compress then Serve: Serving Thousands of LoRA Adapters with Little Overhead}, 
      author={Rickard Brüel-Gabrielsson and Jiacheng Zhu and Onkar Bhardwaj and Leshem Choshen and Kristjan Greenewald and Mikhail Yurochkin and Justin Solomon},
      year={2024},
      eprint={2407.00066},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2407.00066}, 
}
```

### Introduction

This is a work in progress. This repo contains code that is used for compressing LoRA and PEFT adapters, both jointly and individually.

Example of how to use it:

```python 

from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict 

lora_module_list = [
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task280",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task190",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task391",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task290",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1391",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1342",   
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task442",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task620",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1598",
    "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task039"
    ]

model_name_or_path = None

default_peft_model_id = lora_module_list[0]
# find the base model
if model_name_or_path is None:
    model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, # False for pissa
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
)

org_base_model = copy.deepcopy(base_model)
    
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
try:
    # Note that the passed model may be modified inplace.
    peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
except:
    raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')

# this is to have a default model with all the modules present
peft_model = peft_model.to(device)
peft_model.eval()

print("> Begin to load lora modules")
cache = {}

for peft_model_id in tqdm(lora_module_list):
    print("> Loading {} ...".format(peft_model_id))
    cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id).to(device)
    cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))
    del cur_peft_model

## LoRALoLAs

from LoRALoLA.lolas import lola_loras, set_lora_from_dict

# We get the necessary compressed versions
lolas_dict = lola_loras(lora_module_list, cache, r=16, type="diagonal", sparse_reg=0, transform_lora="none")

final_state_dict_lora = set_lora_from_dict(peft_model, lolas_dict, list_lora_module_list, eval_task_to_checkpoints[task_name], project=args.project, type=type)

# set the final weights
set_peft_model_state_dict(peft_model, final_state_dict_lora)

# now you can use the model recreated from the compressed PEFT to see how it performs
```