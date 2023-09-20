
class FeedForwardModel(tf.keras.Model):
    def __init__(self, input_vocab_size, output_vocab_size, max_input_length, max_output_length, embedding_dim, hidden_units):
        super(FeedForwardModel, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        
        # Embedding
        self.embedding_layer = Embedding(input_vocab_size, embedding_dim)
        
        # Feed-forward layers
        self.ff_layer1 = Dense(hidden_units, activation='relu')
        self.ff_layer2 = Dense(hidden_units, activation='relu')
        
        # Output
        self.output_layer = Dense(output_vocab_size, activation='softmax')
    
    def call(self, inputs):
        input_seq, output_seq = inputs
        
        # Embedding
        embedded_input = self.embedding_layer(input_seq)
        embedded_output = self.embedding_layer(output_seq)
        
        # Feed-forward layers
        ff_output1 = self.ff_layer1(embedded_input)
        ff_output2 = self.ff_layer2(embedded_output)
        
        # Output
        output_probs = self.output_layer(ff_output2)
        
        return output_probs