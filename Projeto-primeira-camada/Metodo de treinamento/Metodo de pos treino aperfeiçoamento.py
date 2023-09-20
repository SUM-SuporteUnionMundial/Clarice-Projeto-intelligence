# Importações necessárias
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Função para realizar o treinamento completo com avaliação competitiva
def treinamento_completo_com_avaliacao(modelo, X_train, y_train, X_val, y_val, X_test, y_test):
    # Compilar o modelo para a geração de códigos
    modelo.compile(optimizer=Adam(clipnorm=1.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Criar uma cópia do modelo para a avaliação de códigos gerados
    modelo_avaliador = tf.keras.models.clone_model(modelo)
    modelo_avaliador.compile(optimizer=Adam(clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])

    # Early Stopping para prevenir overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    for epoch in range(50):
        # Treinar o modelo gerador
        historia_gerador = modelo.fit(
            X_train, y_train,
            epochs=1,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        
        # Usar o modelo gerador para gerar códigos
        codigos_gerados = modelo.predict(X_train)
        
        # Treinar o modelo avaliador com os códigos gerados e os códigos reais
        historia_avaliador = modelo_avaliador.fit(
            codigos_gerados, y_train,
            epochs=1,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )

    # Avaliar o modelo gerador no conjunto de teste
    loss, accuracy = modelo.evaluate(X_test, y_test)
    print(f"Acurácia do modelo gerador no conjunto de teste: {accuracy:.4f}")

# Defina o seu modelo único
modelo = tf.keras.Sequential(...)  # Defina o modelo que você deseja usar para gerar e avaliar códigos

# Defina seus dados de treinamento, validação e teste (X_train, y_train, X_val, y_val, X_test, y_test)

# Chame a função de treinamento completo com avaliação competitiva
treinamento_completo_com_avaliacao(modelo, X_train, y_train, X_val, y_val, X_test, y_test)
