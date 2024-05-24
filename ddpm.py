# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(beta_start, beta_end, num_timesteps, dtype = np.float32)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)
alpha_bar_tf = tf.convert_to_tensor(alpha_bar, dtype = tf.float32)

def forward_diffusion_process(x0, t):
    batch_size = tf.shape(x0)[0]
    noise = tf.random.normal(tf.shape(x0))
    sqrt_alpha_bar_t = tf.gather(tf.sqrt(alpha_bar_tf), t)
    sqrt_one_minus_alpha_bar_t = tf.gather(tf.sqrt(1.0 - alpha_bar_tf), t)
    xt = sqrt_alpha_bar_t[:, None, None, None] * x0 + sqrt_one_minus_alpha_bar_t[:, None, None, None] * noise
    return xt, noise

def build_unet_with_timestep(input_shape):
    image_input = layers.Input(shape = input_shape)
    t_input = layers.Input(shape = (), dtype = tf.int32)
    t_embedding = layers.Embedding(input_dim = num_timesteps, output_dim = 64)(t_input)
    t_embedding = layers.Dense(np.prod(input_shape))(t_embedding)
    t_embedding = layers.Reshape(input_shape)(t_embedding)
    x = layers.Concatenate()([image_input, t_embedding])
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    up1 = layers.UpSampling2D((2, 2))(conv3)
    concat1 = layers.Concatenate()([up1, conv2])
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    up2 = layers.UpSampling2D((2, 2))(conv4)
    concat2 = layers.Concatenate()([up2, conv1])
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    outputs = layers.Conv2D(1, (1, 1))(conv5)
    model = models.Model([image_input, t_input], outputs)
    return model

input_shape = (28, 28, 1)  
model = build_unet_with_timestep(input_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')/255                
x_train = np.expand_dims(x_train, axis=-1)  
x_test = x_test.astype('float32')/255                  
x_test = np.expand_dims(x_test, axis=-1)  
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size = 32)    
val_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size = 32)

save_best_model = callbacks.ModelCheckpoint(checkpoint_path,
                                        monitor='val_loss',
                                        mode='min',
                                        save_best_only=True, save_weights_only = True,
                                        verbose=1)
save_best_model.model = model
csv_logger = callbacks.CSVLogger(filename=csv_logger_path, append=True)
csv_file = open(csv_logger_path, 'a')
csv_logger.csv_file = csv_file

def loss_fn(model, x0, t):
    xt, noise = forward_diffusion_process(x0, t)
    noise_pred = model([xt, t], training=True)
    return tf.reduce_mean(tf.keras.losses.MSE(noise, noise_pred))

@tf.function
def train_step(x0, t):
    with tf.GradientTape() as tape:
         loss = loss_fn(model, x0, t)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
num_epochs = 200              
for epoch in range(num_epochs):
    for batch in train_dataset:
        x0 = batch
        t = tf.random.uniform([tf.shape(x0)[0]], minval = 0, maxval = num_timesteps, dtype = tf.int32)
        train_loss = train_step(x0, t)
    val_losses = []
    for batch in val_dataset:
        x0 = batch
        t = tf.random.uniform([tf.shape(x0)[0]], minval=0, maxval=num_timesteps, dtype=tf.int32)
        val_loss = loss_fn(model, x0, t)
        val_losses.append(val_loss)
    val_loss = tf.reduce_mean(val_losses)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss.numpy()}, Val Loss: {val_loss.numpy()}")
    save_best_model.on_epoch_end(epoch, {"val_loss": val_loss.numpy()})
    csv_logger.on_epoch_end(epoch, {"loss": train_loss.numpy(), "val_loss": val_loss.numpy()})

def sample(model, num_samples, num_timesteps):
    x = tf.random.normal((num_samples, 28, 28, 1))
    y = tf.identity(x)
    for t in reversed(range(num_timesteps)):
        t_tensor = tf.convert_to_tensor([t] * num_samples, dtype=tf.int32)
        noise_pred = model([x, t_tensor])
        alpha_t = alphas[t]
        alpha_t_bar = tf.gather(alpha_bar_tf, t)
        alpha_t_minus_one_bar = tf.gather(alpha_bar_tf, t - 1)
        beta_t = betas[t]
        sigma_t = tf.sqrt(beta_t * (1.0 - alpha_t_minus_one_bar) / (1.0 - alpha_t_bar))
        if t > 0:
           noise = tf.random.normal(tf.shape(x))
        else:
           noise = tf.zeros_like(x)
        x = (x - beta_t/tf.sqrt(1.0 - alpha_t_bar) * noise_pred)/tf.sqrt(alpha_t) + noise * sigma_t
    return x, y
