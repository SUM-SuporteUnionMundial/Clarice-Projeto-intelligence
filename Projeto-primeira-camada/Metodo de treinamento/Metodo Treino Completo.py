# Função para realizar o treinamento completo com transferência de aprendizado
# Importações necessárias
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Função para carregar código de um arquivo
def carregar_codigo(caminho):
    with open(caminho, 'r') as arquivo:
        codigo = arquivo.read()
    return codigo

# Carregar o código do modelo a ser treinado
caminho_arquivo = "/caminho/para/arquivo.py"
codigo_modelo = carregar_codigo(caminho_arquivo)
modelo_virgem = codigo_modelo

# Defina a classe do modelo de agregação
class NeuralAggregationCombinedModel(nn.Module):
    def __init__(self, models):
        super(NeuralAggregationCombinedModel, self).__init__()
        self.models = models
        self.aggregation_layer = nn.Sequential(
            nn.Linear(len(models), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        outputs = [model(**inputs).logits for model in self.models]
        aggregated_outputs = torch.cat(outputs, dim=-1)
        aggregated_outputs = self.aggregation_layer(aggregated_outputs)
        return aggregated_outputs

# Função para carregar modelos pré-treinados
def carregar_modelos_pre_treinados(caminhos_dos_modelos):
    modelos = []
    for caminho_do_modelo in caminhos_dos_modelos:
        config = AutoConfig.from_pretrained(caminho_do_modelo)
        modelo = AutoModel.from_pretrained(caminho_do_modelo, config=config)
        modelos.append(modelo)
    return modelos

# Lista de caminhos para os modelos pré-treinados
caminhos_dos_modelos = [
    "/caminho/para/modelo1",
    "/caminho/para/modelo2",
    "/caminho/para/modelo3"
]

# Carregar modelos pré-treinados
modelos_pretreinados = carregar_modelos_pre_treinados(caminhos_dos_modelos)

# Criar a instância do modelo de agregação
modelo_combinado = NeuralAggregationCombinedModel(modelos_pretreinados)

# Função para transferir o aprendizado dos modelos unidos para o modelo virgem
def transferir_aprendizado(modelo_virgem, modelo_combinado):
    for layer_virgem, layer_combinado in zip(modelo_virgem.layers[:-3], modelo_combinado.layers[:-3]):
        layer_virgem.set_weights(layer_combinado.get_weights())
        layer_virgem.trainable = False
# Chame a função de transferência de aprendizado
transferir_aprendizado(modelo_virgem, modelo_combinado)

# Compilar o modelo virgem
modelo_virgem.compile(optimizer=Adam(clipnorm=1.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def transferencia_aprendizado(modelo_virgem, modelos_pre_treinados):
    # Calcula o número de camadas a serem congeladas nos modelos pré-treinados
    num_camadas_congeladas = int(0.75 * len(modelos_pre_treinados[0].layers))
    
    # Congela as primeiras 'num_camadas_congeladas' camadas de cada modelo pré-treinado
    for modelo in modelos_pre_treinados:
        for layer in modelo.layers[:num_camadas_congeladas]:
            layer.trainable = False
    
    # Adiciona todas as camadas dos modelos pré-treinados ao modelo virgem
    for modelo in modelos_pre_treinados:
        for layer in modelo.layers:
            modelo_virgem.add(layer)
    
    return modelo_virgem

# Chamar a função para transferir o aprendizado
modelo_transferido = transferencia_aprendizado(modelo_virgem, modelos_pre_treinados)


# Função para realizar o treinamento completo com transferência de aprendizado
def treinamento_completo(modelo_transferido, X_train, y_train, X_val, y_val, X_test, y_test):
    # Compilar o modelo transferido
    modelo_transferido.compile(optimizer=Adam(clipnorm=1.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Early Stopping para prevenir overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Treinar o modelo transferido
    history = modelo_transferido.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Avaliar o modelo transferido no conjunto de teste
    loss, accuracy = modelo_transferido.evaluate(X_test, y_test)
    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# Defina seus dados de treinamento, validação e teste (X_train, y_train, X_val, y_val, X_test, y_test)

# Chame a função de treinamento completo
treinamento_completo(modelo_transferido, X_train, y_train, X_val, y_val, X_test, y_test)