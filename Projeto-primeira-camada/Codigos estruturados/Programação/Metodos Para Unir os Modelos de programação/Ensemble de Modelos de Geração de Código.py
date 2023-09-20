class CodeGenerationEnsembleLayer(tf.keras.layers.Layer):
    def __init__(self, models, **kwargs):
        super(CodeGenerationEnsembleLayer, self).__init__(**kwargs)
        self.models = models
    
    def call(self, inputs):
        outputs = []
        
        # Gerar saídas de cada modelo
        for model in self.models:
            output_probs = model(inputs)  # Passar as mesmas entradas para cada modelo
            outputs.append(output_probs)
        
        # Média das probabilidades de saída dos modelos
        combined_probs = tf.reduce_mean(outputs, axis=0)
        
        return combined_probs

#exemplo de uso Para usar essa classe, você pode instanciá-la da seguinte maneira

# Criar instâncias dos modelos
model1 = RecurrentCodeGenerator(...)
model2 = SimpleRecurrentCodeGenerator(...)
model3 = FeedForwardModel(...)
model4 = ConvolutionalModel(...)
model5 = DualLSTMModel(...)
model6 = NPIModel(...)
model7 = NPSModel(...)
model8 = BiLSTMModel(...)

# Criar uma instância da camada Ensemble
ensemble_layer = CodeGenerationEnsembleLayer([model1, model2, model3, model4, model5, model6, model7, model8])

# Entrada para a geração de código (substitua pelo dado real)
input_data = ...

# Gerar saída combinada usando a camada Ensemble
combined_output_probs = ensemble_layer(input_data)