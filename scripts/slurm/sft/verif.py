from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    '/leonardo_work/OELLM_prod2026/ytahtah0/models/OLMo-3-7B-Instruct-SFT-reproduce',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('/leonardo_work/OELLM_prod2026/ytahtah0/models/OLMo-3-7B-Instruct-SFT-reproduce')

messages = [{'role': 'user', 'content': 'What is 25 * 37?'}]
inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.6, top_p=0.95, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
