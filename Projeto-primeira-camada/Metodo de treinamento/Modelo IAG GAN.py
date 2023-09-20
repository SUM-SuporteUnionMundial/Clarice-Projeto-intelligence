import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D

def criar_modelo_generativo():
    gerador = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(784, kernel_size=(3, 3), activation='sigmoid')
    ])
    return gerador

def criar_discriminador():
    discriminador = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return discriminador

def criar_gan(gerador, discriminador):
    gan = tf.keras.Model(gerador.input, discriminador(gerador.output))
    perda_discriminador = tf.keras.losses.BinaryCrossentropy()
    perda_gerador = tf.keras.losses.BinaryCrossentropy()
    otimizador_discriminador = tf.keras.optimizers.Adam(learning_rate=0.001)
    otimizador_gerador = tf.keras.optimizers.Adam(learning_rate=0.001)
    return gan, perda_discriminador, perda_gerador, otimizador_discriminador, otimizador_gerador

# Criar o gerador e o discriminador
gerador_D = criar_modelo_generativo()
discriminador_D = criar_discriminador()

# Criar o GAN
gan_D, perda_discriminador_D, perda_gerador_D, otimizador_discriminador_D, otimizador_gerador_D = criar_gan(gerador_D, discriminador_D)

