import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('data/540-modified.csv', sep=';')
print(df.shape)
df.head()

#drop null IST
df = df[df['Interstitial glucose'].notna()]

device_time = df.pop('Device Time')
#df.pop('Sleep quality')


n = len(df)
mean = df.mean()
std = df.std()

"""Get data where IST is null"""

# get IST only
ist = df.pop('Interstitial glucose')
carb =  df.pop('Carbohydrate intake')
df = pd.DataFrame(data={'Interstitial glucose' : ist, 'Carbohydrate intake' : carb})
df.head()

#clean data
#replace NaN values
df_clean = df.fillna(0)
df_clean.head()
#replace negative values
df_clean[df_clean < 0] = 0

#uncomment to work with not NaN values
df = df_clean
df.head()

#get timestamp and weekday
date_time = pd.to_datetime(device_time, format='%Y-%m-%d %H:%M:%S')
time = date_time.apply(lambda date : date.time())
df['timestamp'] = time.apply(lambda t: (t.hour * 60 + t.minute) * 60 + t.second)

df['weekday'] = date_time.apply(lambda date: date.weekday())

df.head()

column_indices = {name: i for i, name in enumerate(df.columns)}
print(column_indices)

#split dataframe
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_df.describe().transpose()
# train_df.head()

#normalize
# train_mean = train_df.mean()
# train_std = train_df.std()

# train_df = (train_df - train_mean) / train_std
# val_df = (val_df - train_mean) / train_std
# test_df = (test_df - train_mean) / train_std

#window of consecutive samples from the data
class WindowGenerator():
  def __init__(self, input_width, label_width, shift, 
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    if shift >= 0:
      self.total_window_size = input_width + shift
    else:
      self.total_window_size = input_width

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=32,)
    
    ds = ds.map(self.split_window)
    return ds
    
  def plot(self, model=None, plot_col='Carbohydrate intake', title=None, max_subplots=1, show=False):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(3, 1, n+1)
      plt.ylabel(f'{plot_col}')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
               label='Inputs', marker='.', zorder=-10)
  
      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index
  
      if label_col_index is None:
        continue
  
      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  marker='o', edgecolors='k', label='Labels', c='#2ca02c', s=10)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='^', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=10)
  
      if n == 0:
        plt.legend()
  
    plt.xlabel('t')
    plt.title(title)
    
    if show:
      plt.show()
    
  @property
  def train(self):
    return self.make_dataset(self.train_df)
  
  @property
  def val(self):
    return self.make_dataset(self.val_df)
  
  @property
  def test(self):
    return self.make_dataset(self.test_df)
  
  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

WINDOW_WIDTH =12 #1 hour window
window = WindowGenerator(input_width=WINDOW_WIDTH, label_width=1, shift=0, label_columns=['Carbohydrate intake']) #shift= 0 - label last row; 1 - prediction 1 row; -1 ?

print(window.train.element_spec)
for example_inputs, example_labels in window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
#window.plot()
# window.plot(plot_col='Electrodermal activity')
# window.plot(plot_col='Interstitial glucose')

wide_window = WindowGenerator(input_width=WINDOW_WIDTH*24, label_width=WINDOW_WIDTH*24, shift=0, label_columns=['Carbohydrate intake'])
print(wide_window.train.element_spec)
#wide_window.plot()

MAX_EPOCHS = 20
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
val_performance = {}
performance = {}

linear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

linear_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
linear_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])

val_performance['Linear'] = linear_model.evaluate(window.val)
performance['Linear'] = linear_model.evaluate(window.test)

wide_window.plot(linear_model, title='Linear')

# show linear model weights
#plt.bar(x = range(len(train_df.columns)),
#        height=linear_model.layers[0].kernel[:,0].numpy())
#axis = plt.gca()
#axis.set_xticks(range(len(train_df.columns)))
#_ = axis.set_xticklabels(train_df.columns, rotation=90)

dense_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

dense_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
dense_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])

val_performance['Dense'] = dense_model.evaluate(window.val)
performance['Dense'] = dense_model.evaluate(window.test)

#wide_window.plot(dense_model)

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

multi_step_dense.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
multi_step_dense.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])

#val_performance['MultiStepDense'] = multi_step_dense.evaluate(window.val)
#performance['MultiStepDense'] = multi_step_dense.evaluate(window.test)

#window.plot(multi_step_dense)

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(WINDOW_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

conv_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
conv_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])

val_performance['Conv'] = conv_model.evaluate(window.val)
performance['Conv'] = conv_model.evaluate(window.test)

# window.plot(conv_model)
conv_window = WindowGenerator(input_width=WINDOW_WIDTH*24+WINDOW_WIDTH-1, label_width=WINDOW_WIDTH*24, shift=0, label_columns=['Carbohydrate intake'])
conv_window.plot(conv_model, title='Conv')

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

lstm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
lstm_model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])

val_performance['LSTM'] = lstm_model.evaluate(window.val)
performance['LSTM'] = lstm_model.evaluate(window.test)

wide_window.plot(lstm_model, title='LSTM')

for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.figure(figsize=(12, 8))
plt.ylabel('mean_absolute_error')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()