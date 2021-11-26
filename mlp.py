import tensorflow as tf
from tensorflow import keras
import numpy as np

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


#
# class CustomModel(keras.Model):
#     def train_step(self, data):
#         x, y = data
#
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             loss = keras.losses.mean_squared_error(y, y_pred)
#
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#
#         loss_tracker.update_state(loss)
#         mae_metric.update_state(y, y_pred)
#         return {"loss": loss_tracker.result(), "mae": mae_metric.result()}
#
#     @property
#     def metrics(self):
#         return [loss_tracker, mae_metric]
#
#
# def create_model_and_fit(samples):
#     inputs = keras.Input(20, input_shape=(12,))
#     outputs = keras.layers.Dense(1)(inputs)
#     model = CustomModel(inputs, outputs)
#
#     model.compile(optimizer="adam")
#
#     const_params = np.array([samples['ir'], samples['or'], samples['r_groove'],
#                              samples['r_neck'], samples['t_shoulder'], samples['h_inner'],
#                              samples['h_shoulder'], samples['h_groove'], samples['h_neck'],
#                              samples['h']])
#     x = np.array([])
#     for rx in samples['rx']:
#         # tmp_sample = const_params
#         # tmp_sample.append(rx)
#         new_input = np.append(const_params, [rx], axis=0)
#         x = np.append(x, new_input, axis=0)
#         # x.append(tmp_sample)
#     # y = np.array(samples['er'])
#     y = np.array(samples['er'])
#
#     print(x)
#     print(y)
#     model.fit(x, y, epochs=5)


def create_model_and_fit(samples):
    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(11,)))
    model.add(keras.layers.Dense(20, activation='elu'))
    model.add(keras.layers.Dense(20, activation='elu'))
    model.add(keras.layers.Dense(20, activation='elu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    const_params = np.array([[samples['ir'], samples['or'], samples['r_groove'],
                              samples['r_neck'], samples['t_shoulder'], samples['h_inner'],
                              samples['h_shoulder'], samples['h_groove'], samples['h_neck'],
                              samples['h']]] * len(samples['rx']))

    rx = np.array(samples['rx'])
    x = np.concatenate((const_params, rx[:, None]), axis=1)
    y = np.array(samples['er'])

    model.fit(x, y, epochs=100, verbose=1)
