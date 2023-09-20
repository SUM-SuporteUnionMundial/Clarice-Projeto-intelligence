import tensorflow as tf
from transformers import MarianMTModel, MarianTokenizer

# Dados de exemplo
input_text = "This is an example sentence to be translated."

# Carregue um modelo de tradução
model_name = "Helsinki-NLP/opus-mt-en-pt"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Pré-processe os dados
inputs = tokenizer(input_text, return_tensors="pt")

# Faça a tradução
translation = model.generate(**inputs)
translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

print("Tradução:", translated_text)
