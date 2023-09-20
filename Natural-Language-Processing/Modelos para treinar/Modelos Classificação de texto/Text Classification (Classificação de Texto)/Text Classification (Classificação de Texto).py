import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dados de exemplo
X_train = ...
y_train = ...

# Crie um modelo de classificação de texto
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Treine o modelo
model.fit(X_train, y_train, epochs=5)
