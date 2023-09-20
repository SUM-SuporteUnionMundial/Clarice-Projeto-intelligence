import tensorflow as tf
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Dados de exemplo
input_text = "Este é um exemplo de um texto longo que precisa ser resumido."

# Carregue um modelo de sumarização
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Pré-processe os dados
inputs = tokenizer.encode("sumarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

# Gere a sumarização
summary_ids = model.generate(inputs)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Sumarização:", summary)
