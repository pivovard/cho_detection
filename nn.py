import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#window of consecutive samples from the data
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               data, headers, label_columns=None):

    # Store the raw data.
    df = pd.DataFrame()
    for i, h in enumerate(headers):
      df[h] = data[h]

    #split dataframe
    n = len(df)
    self.train_df = df[0:int(n*0.7)]
    self.val_df = df[int(n*0.7):int(n*0.9)]
    self.test_df = df[int(n*0.9):]

    self.datetime = data['datetime']

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.column_indices = {name: i for i, name in enumerate(self.train_df.columns)}

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
      # plt.plot(self.input_indices, inputs[n, :, plot_col_index],
      #          label='Inputs', marker='.', zorder=-10)
      plt.plot(self.datetime[n:(n+1)*self.total_window_size], inputs[n, :, plot_col_index],
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