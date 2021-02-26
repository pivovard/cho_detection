import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import timedelta

from wg import WindowGenerator
import utils

MAX_EPOCHS = 20

headers = [utils.ist_l, utils.inr_l, utils.inb_l, 'hour', 'weekday']
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(5) #num features

  def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

  def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the lstm state
    prediction, state = self.warmup(inputs)

    # Insert the first prediction
    predictions.append(prediction)

    # Run the rest of the prediction steps
    for n in range(1, self.out_steps):
      # Use the last prediction as input.
      x = prediction
      # Execute one lstm step.
      x, state = self.lstm_cell(x, states=state,
                                training=training)
      # Convert the lstm output to a prediction.
      prediction = self.dense(x)
      # Add the prediction to the output
      predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions

def single_step(df):
	window = WindowGenerator(data=df, headers=headers,
								input_width=utils.WINDOW_WIDTH_1H, label_width=1, shift=3,
								label_columns=['Interstitial glucose'])
	window1 = WindowGenerator(data=df, headers=headers,
								input_width=utils.WINDOW_WIDTH_1H, label_width=utils.WINDOW_WIDTH_1H, shift=3,
								label_columns=['Interstitial glucose'])  
	window24 = WindowGenerator(data=df, headers=headers,
								input_width=utils.WINDOW_WIDTH_24H, label_width=utils.WINDOW_WIDTH_24H, shift=3,
								label_columns=['Interstitial glucose'])
	
	lstm_model = tf.keras.models.Sequential([
		# Shape [batch, time, features] => [batch, time, lstm_units]
		tf.keras.layers.LSTM(32, return_sequences=True),
		# Shape => [batch, time, features]
		tf.keras.layers.Dense(units=1)
	])
	
	lstm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
	lstm_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[])
	
	#window.plot(lstm_model, title='LSTM')
	window1.plot(lstm_model, title='LSTM')
	#window24.plot(lstm_model, title='LSTM')
	plt.show()

def multi_step(df):
  num_features = 5

  window = WindowGenerator(data=df, headers=headers, label_columns=['Interstitial glucose'],
                              input_width=utils.WINDOW_WIDTH_1H*2, label_width=utils.WINDOW_WIDTH_1H, shift=utils.WINDOW_WIDTH_1H)
  window24 = WindowGenerator(data=df, headers=headers, label_columns=['Interstitial glucose'],
                              input_width=utils.WINDOW_WIDTH_24H, label_width=utils.WINDOW_WIDTH_1H, shift=utils.WINDOW_WIDTH_1H)

  dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(utils.WINDOW_WIDTH_1H*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([utils.WINDOW_WIDTH_1H, num_features])
  ])

  dense_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
  dense_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[])
	
  window.plot(dense_model, title='Dense')
  window24.plot(dense_model, title='Dense')

  CONV_WIDTH = 3
  conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(utils.WINDOW_WIDTH_1H*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([utils.WINDOW_WIDTH_1H, num_features])
  ])

  conv_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
  conv_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[])

  window.plot(conv_model, title='Conv')
  window24.plot(conv_model, title='Conv')

  lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(utils.WINDOW_WIDTH_1H*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([utils.WINDOW_WIDTH_1H, num_features])
  ])
  
  lstm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
  lstm_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[])
	
  window.plot(lstm_model, title='LSTM')
  window24.plot(lstm_model, title='LSTM')

  plt.show()

def feedback(window):
  utils.printh('Building RNN model.')

  feedback_model = FeedBack(units=32, out_steps=6)
  prediction, state = feedback_model.warmup(window.example_train[0])

  feedback_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
  feedback_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[])
	
  feedback_model.summary()

  window.plot(feedback_model, title='LSTM')
  plt.show()

  return feedback_model

def predict(model, window):
  utils.printh('Test data prediction.')

  predictions = pd.DataFrame(columns=['label', 't1', 't2', 't3', 't4', 't5', 't6'], dtype=float)
  it = iter(window.test)
  for i in range(len(window.test)):
    inputs, labels = next(it)
    pred = model(inputs)

    for j in range(pred.shape[0]):
      predictions.loc[i*32+j] = np.insert(pred[j, :, 0], 0, labels[j,0,0])
  
  # append empty rows for shifting
  # predictions = predictions.append(pd.Series(), ignore_index=True)
  predictions = predictions.append(pd.DataFrame([[np.nan for i in predictions.columns] for i in range(5)], columns=predictions.columns))
  # shift columns
  predictions['t2']= predictions['t2'].shift(1)
  predictions['t3']= predictions['t3'].shift(2)
  predictions['t4']= predictions['t4'].shift(3)
  predictions['t5']= predictions['t5'].shift(4)
  predictions['t6']= predictions['t6'].shift(5)
  print(predictions.head(10))

  datetime = window.datetime[int(len(window.datetime)*0.9)+utils.WINDOW_WIDTH_1H*3:]

  fig = plt.figure(figsize=(12, 8))
  plt.subplot(6, 1, 1)
  plt.plot(datetime, predictions['label'], label='ist')
  plt.plot(datetime, predictions['t1'], label='t1')
  plt.legend()
  plt.subplot(6, 1, 2)
  plt.plot(datetime, predictions['label'], label='ist')
  plt.plot(datetime, predictions['t2'], label='t2')
  plt.legend()
  plt.subplot(6, 1, 3)
  plt.plot(datetime, predictions['label'], label='ist')
  plt.plot(datetime, predictions['t3'], label='t3')
  plt.legend()
  plt.subplot(6, 1, 4)
  plt.plot(datetime, predictions['label'], label='ist')
  plt.plot(datetime, predictions['t4'], label='t4')
  plt.legend()
  plt.subplot(6, 1, 5)
  plt.plot(datetime, predictions['label'], label='ist')
  plt.plot(datetime, predictions['t5'], label='t5')
  plt.legend()
  plt.subplot(6, 1, 6)
  plt.plot(datetime, predictions['label'], label='ist')
  plt.plot(datetime, predictions['t6'], label='t6')
  plt.legend()
  plt.show()

  # utils.printh('Derivations of predicted values')
  # der = pd.DataFrame(columns=['d1', 'd2', 'd3'], dtype=float)
  # for i in range(len(predictions)-6):
  #   d1=(predictions.loc[i+1,'t1']-predictions.loc[i,'label'])/5
  #   d2=d1/5
  #   d3=d2/5
  #   der.loc[i] = [d1,d2,d3]
  
  # fig = plt.figure(figsize=(12, 8))
  # plt.subplot(2, 1, 1)
  # plt.title('Derivations')
  # plt.plot(datetime[1:-5], predictions['label'][1:-5], label='ist')
  # plt.legend()
  # plt.subplot(2, 1, 2)
  # plt.plot(datetime[1:-5], der['d1'], label='d1')
  # plt.plot(datetime[1:-5], der['d2'], label='d2')
  # plt.plot(datetime[1:-5], der['d3'], label='d3')
  # plt.legend()
  # plt.show()