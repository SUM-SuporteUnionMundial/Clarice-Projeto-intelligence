import transformers

# Dados de exemplo
input_text = "O [MASK] é a capital da França."

# Carregue um modelo BERT
model_name = "bert-base-multilingual-cased"
model = transformers.BertForMaskedLM.from_pretrained(model_name)
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

# Pré-processe os dados
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Faça a previsão para os tokens mascarados
with torch.no_grad():
    predictions = model(inputs).logits

predicted_token_id = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.decode(predicted_token_id)

print("Palavra Preenchida:", predicted_token)
