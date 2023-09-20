import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM, AdditiveAttention, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Model

class CodeStyleTransferModel(tf.keras.Model):
    def __init__(self, max_text_length, vocab_size, embedding_dim, hidden_dim, memory_slots, model_type):
        super(CodeStyleTransferModel, self).__init__()
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        self.model_type = model_type
        
        self.embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        if model_type == 'RecurrentCodeGenerator':
            self.code_generator = RecurrentCodeGenerator(max_text_length, vocab_size, embedding_dim, hidden_dim)
        elif model_type == 'SimpleRecurrentCodeGenerator':
            self.code_generator = SimpleRecurrentCodeGenerator(max_text_length, vocab_size, embedding_dim, hidden_dim)
        elif model_type == 'FeedForwardModel':
            self.code_generator = FeedForwardModel(vocab_size, vocab_size, max_text_length, max_text_length, embedding_dim, hidden_dim)
        elif model_type == 'ConvolutionalModel':
            self.code_generator = ConvolutionalModel(vocab_size, vocab_size, max_text_length, max_text_length, embedding_dim, hidden_dim, 3)  # 3 is kernel size
        elif model_type == 'DualLSTMModel':
            self.code_generator = DualLSTMModel(vocab_size, vocab_size, max_text_length, max_text_length, embedding_dim, hidden_dim)
        elif model_type == 'BiLSTMModel':
            self.code_generator = BiLSTMModel(vocab_size, vocab_size, max_text_length, max_text_length, embedding_dim, hidden_dim)
        elif model_type == 'NPSModel':
            self.code_generator = NPSModel(vocab_size, vocab_size, max_text_length, max_text_length, embedding_dim, hidden_dim)
        elif model_type == 'NPIModel':
            self.code_generator = NPIModel(vocab_size, vocab_size, max_text_length, max_text_length, embedding_dim, hidden_dim, memory_slots)
        else:
            raise ValueError("Invalid model_type")
        
        self.attention_layer = AdditiveAttention()
        self.output_layer = Dense(vocab_size, activation='softmax')
        
    def call(self, inputs):
        text_input = inputs
        
        embedded = self.embedding_layer(text_input)
        code_gen_output = self.code_generator(embedded)
        
        attention_output = self.attention_layer([code_gen_output, code_gen_output])
        output_probs = self.output_layer(attention_output)
        
        return output_probs
        
#exemplo de uso Para usar essa classe, você pode instanciá-la da seguinte maneira
max_text_length = ...
vocab_size = ...
embedding_dim = ...
hidden_dim = ...
memory_slots = ...
model_type = 'RecurrentCodeGenerator'  # Escolha o tipo de modelo que você deseja criar

code_style_transfer_model = CodeStyleTransferModel(max_text_length, vocab_size, embedding_dim, hidden_dim, memory_slots, model_type)