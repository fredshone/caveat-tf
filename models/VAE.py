
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Layer, Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.utils import plot_model

from utils.callbacks import CustomCallback, step_decay_schedule 

import numpy as np
import json
import os
import pickle


class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):

    def __init__(
        self,
        input_dim,
        encoder_conv_filters,
        encoder_conv_kernel_size,
        encoder_conv_strides,
        decoder_conv_t_filters,
        decoder_conv_t_kernel_size,
        decoder_conv_t_strides,
        z_dim,
        use_batch_norm = False,
        use_dropout= False,
        **kwargs,
        ):

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)
        self.shape_before_flattening = None

        self.encoder = None
        self.decoder = None
        self.mode = None
        self._build()

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.reconstruction_weight = 1000

    def compile(
        self,
        learning_rate=0.0005,
        reconstruction_weight=1000,
         **kwargs
         ):
        super().compile(**kwargs)
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def fit(
        self,
        x_train,
        batch_size,
        epochs,
        run_folder,
        print_every_n_batches = 100,
        initial_epoch = 0,
        lr_decay = 1,
        shuffle = True
    ):
        callbacks_list = self._callbacks(
            run_folder = run_folder,
            print_every_n_batches = print_every_n_batches,
            lr_decay = lr_decay
    )

        super().fit(
            x_train,
            batch_size = batch_size,
            shuffle = shuffle,
            epochs = epochs,
            initial_epoch = initial_epoch,
            callbacks = callbacks_list,
    )

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.reconstruction_weight * tf.reduce_mean(
                tf.square(data - reconstruction), axis = (1,2,3)
                )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.encoder_conv_filters
                , self.encoder_conv_kernel_size
                , self.encoder_conv_strides
                , self.decoder_conv_t_filters
                , self.decoder_conv_t_kernel_size
                , self.decoder_conv_t_strides
                , self.z_dim
                , self.use_batch_norm
                , self.use_dropout
                ], f)

        # self.plot_model(folder)

    def save_weights(self, path, **kwargs):
        self.model.save_weights(path, **kwargs)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)

    def _build(self):
        
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters = self.encoder_conv_filters[i]
                , kernel_size = self.encoder_conv_kernel_size[i]
                , strides = self.encoder_conv_strides[i]
                , padding = 'same'
                , name = 'encoder_conv_' + str(i)
                )
            x = conv_layer(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate = 0.25)(x)

        self.shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)
        # self.encoder_mu_log_var = keras.Model(encoder_input, (self.mu, self.log_var))
        encoder_output = Sampling(name='encoder_output')([self.mu, self.log_var])
        self.encoder = keras.Model(encoder_input, [self.mu, self.log_var, encoder_output], name='encoder')
        # self.encoder.summary()

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        x = Dense(np.prod(self.shape_before_flattening))(decoder_input)
        x = Reshape(self.shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i]
                , kernel_size = self.decoder_conv_t_kernel_size[i]
                , strides = self.decoder_conv_t_strides[i]
                , padding = 'same'
                , name = 'decoder_conv_t_' + str(i)
                )
            x = conv_t_layer(x)
            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        self.decoder_output = x
        self.decoder = keras.Model(decoder_input, self.decoder_output, name="decoder")
        # self.decoder.summary()

        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        self.model = keras.Model(model_input, model_output)

    def _callbacks(
        self,
        run_folder,
        print_every_n_batches = 100,
        initial_epoch = 0,
        lr_decay = 1
        ):

        custom_callback = CustomCallback(
            run_folder,
            print_every_n_batches,
            initial_epoch,
            self
            )
        lr_sched = step_decay_schedule(
            initial_lr=self.learning_rate,
            decay_factor=lr_decay,
            step_size=1
            )
        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(
            checkpoint_filepath,
            save_weights_only = True,
            verbose=1
            )
        checkpoint2 = ModelCheckpoint(
            os.path.join(run_folder, 'weights/weights.h5'),
            save_weights_only = True,
            verbose=1
            )
        return [checkpoint1, checkpoint2, custom_callback, lr_sched]
