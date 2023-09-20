
class NPSModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, max_input_length, max_output_length, embedding_dim, hidden_units):
        super(NPSModel, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        
        # Encoder
        self.encoder_embedding = Embedding(input_vocab_size, embedding_dim)
        self.encoder_lstm_cell = LSTMCell(hidden_units)
        self.encoder_rnn = RNN(self.encoder_lstm_cell, return_sequences=True, return_state=True)
        
        # Decoder
        self.decoder_embedding = Embedding(output_vocab_size, embedding_dim)
        self.decoder_lstm_cell = LSTMCell(hidden_units)
        self.decoder_rnn = RNN(self.decoder_lstm_cell, return_sequences=True, return_state=True)
        
        # Attention
        self.attention = AdditiveAttention()
        
        # Output
        self.output_layer = Dense(output_vocab_size, activation='softmax')
    
    def call(self, inputs):
        input_seq, output_seq = inputs
        
        # Encoder
        encoder_embedded = self.encoder_embedding(input_seq)
        encoder_outputs, encoder_state_h, encoder_state_c = self.encoder_rnn(encoder_embedded)
        
        # Decoder
        decoder_embedded = self.decoder_embedding(output_seq)
        decoder_outputs, _, _ = self.decoder_rnn(decoder_embedded, initial_state=[encoder_state_h, encoder_state_c])
        
        # Attention
        attention_outputs = self.attention([decoder_outputs, encoder_outputs])
        
        # Output
        output_probs = self.output_layer(attention_outputs)
        
        return output_probs
