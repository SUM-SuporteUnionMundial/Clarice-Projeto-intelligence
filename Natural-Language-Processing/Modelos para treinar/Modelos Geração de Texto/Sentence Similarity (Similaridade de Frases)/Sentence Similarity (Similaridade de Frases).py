import tensorflow as tf
from tensorflow.keras import layers

# Dados de exemplo
sentence1 = "Este é um exemplo de frase."
sentence2 = "Esta é outra frase para comparação."

# Crie um modelo de similaridade de frases usando Siamese Network
input_layer = layers.Input(shape=(max_sequence_length,), dtype="int32")
embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=128)(input_layer)
lstm_layer = layers.LSTM(64)(embedding_layer)

model = keras.Model(inputs=input_layer, outputs=lstm_layer)

# Pré-processe os dados
tokenized_sentence1 = tokenizer(sentence1, return_tensors="tf")["input_ids"]
tokenized_sentence2 = tokenizer(sentence2, return_tensors="tf")["input_ids"]

# Calcule a similaridade usando o modelo
embedding_sentence1 = model.predict(tokenized_sentence1)
embedding_sentence2 = model.predict(tokenized_sentence2)

similarity_score = cosine_similarity(embedding_sentence1, embedding_sentence2)
print("Similaridade:", similarity_score)
