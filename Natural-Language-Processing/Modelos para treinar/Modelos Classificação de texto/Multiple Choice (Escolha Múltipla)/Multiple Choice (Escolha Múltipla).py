import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dados de exemplo
question = "Qual é a capital da França?"
options = ["Paris", "Londres", "Berlim", "Madri"]
correct_answer = "Paris"

# Crie um modelo de classificação para escolha múltipla
input_text = [question + " " + option for option in options]
tokenized_texts = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True, max_length=128)
labels = [1 if option == correct_answer else 0 for option in options]

model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treine o modelo
model.fit(tokenized_texts, labels, epochs=5)
