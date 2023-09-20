import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer

# Dados de exemplo
context = "Exemplo de tabela..."
question = "Qual é o valor da coluna X para a linha Y?"

# Carregue um modelo para perguntas e respostas
model_name = "distilbert-base-cased-distilled-squad"
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Pré-processe os dados
inputs = tokenizer(context, question, return_tensors="tf")

# Faça a previsão
outputs = model(inputs)
answer_start = tf.argmax(outputs.start_logits, axis=1)
answer_end = tf.argmax(outputs.end_logits, axis=1)
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+1]))

print("Resposta:", answer)
