
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM, AdditiveAttention, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model

def create_attention_based_code_generator(max_text_length, vocab_size, embedding_dim, hidden_dim, model_type):
    text_input = Input(shape=(max_text_length,), dtype='int32')
    
    if model_type == 'Recurrent':
        model = RecurrentCodeGenerator(max_text_length, vocab_size, embedding_dim, hidden_dim)(text_input)
    elif model_type == 'SimpleRecurrent':
        model = SimpleRecurrentCodeGenerator(max_text_length, vocab_size, embedding_dim, hidden_dim)(text_input)
    elif model_type == 'FeedForward':
        model = FeedForwardModel(max_text_length, vocab_size, embedding_dim, hidden_dim)(text_input)
    elif model_type == 'Convolutional':
        model = ConvolutionalModel(max_text_length, vocab_size, embedding_dim, hidden_dim)(text_input)
    elif model_type == 'DualLSTM':
        model = DualLSTMModel(max_text_length, vocab_size, embedding_dim, hidden_dim)(text_input)
    elif model_type == 'NPS':
        model = NPSModel(max_text_length, vocab_size, embedding_dim, hidden_dim)(text_input)
    elif model_type == 'NPI':
        model = NPIModel(max_text_length, vocab_size, embedding_dim, hidden_dim, memory_slots)(text_input)
    else:
        raise ValueError("Invalid model_type")
    
    attention_layer = AdditiveAttention()([model, model])
    output_probs = Dense(vocab_size, activation='softmax')(attention_layer)
    
    model = Model(inputs=text_input, outputs=output_probs)
    return model

#exemplo de uso Para usar essa classe, você pode instanciá-la da seguinte maneira
max_text_length = ...
vocab_size = ...
embedding_dim = ...
hidden_dim = ...
memory_slots = ...

model_type = 'Recurrent'  # Escolha o tipo de modelo que você deseja criar

attention_based_model = create_attention_based_code_generator(max_text_length, vocab_size, embedding_dim, hidden_dim, model_type)

