
class NPIModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, max_input_length, max_output_length, embedding_dim, hidden_units, memory_slots):
        super(NPIModel, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.memory_slots = memory_slots
        
        # Encoder
        self.encoder_embedding = Embedding(input_vocab_size, embedding_dim)
        self.encoder_lstm_cell = LSTMCell(hidden_units)
        self.encoder_rnn = RNN(self.encoder_lstm_cell, return_sequences=True, return_state=True)
        
        # External Memory
        self.memory_attention = AdditiveAttention()
        
        # Decoder
        self.decoder_embedding = Embedding(output_vocab_size, embedding_dim)
        self.decoder_lstm_cell = LSTMCell(hidden_units + memory_slots)  # Concatenate memory with hidden state
        self.decoder_rnn = RNN(self.decoder_lstm_cell, return_sequences=True)
        
        # Composition
        self.composition_layer = Dense(output_vocab_size)
    
    def call(self, inputs):
        input_seq, output_seq = inputs
        
        # Encoder
        encoder_embedding = Embedding(input_vocab_size, embedding_dim)(input_seq)
        encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
        encoder_outputs, _, encoder_state_c = encoder_lstm(encoder_embedding)
        

        # External Memory
        memory = self.memory_attention([encoder_outputs, encoder_outputs])  # Simplified memory mechanism
        
        # Decoder
        decoder_embedded = self.decoder_embedding(output_seq)
        decoder_outputs = self.decoder_rnn(decoder_embedded, initial_state=[tf.concat([memory, encoder_state_c], axis=-1)])  # Concatenate memory with cell state
        
        # Composition
        output_probs = self.composition_layer(decoder_outputs)
        
        return output_probs