import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification

# Dados de exemplo
inputs = ...
labels = ...

# Carregue o tokenizer BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Crie um modelo de classificação por token
model = TFBertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Treine o modelo
model.fit(inputs, labels, epochs=5)
