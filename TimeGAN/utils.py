"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

## Necessary Packages
import numpy as np
import tensorflow as tf


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
  """Basic RNN Cell.
    
  Args:
    - module_name: gru, lstm, or lstmLN
    
  Returns:
    - rnn_cell: RNN Cell
  """
  assert module_name in ['gru','lstm','lstmLN']
  
  # GRU
  if (module_name == 'gru'):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM
  elif (module_name == 'lstm'):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM Layer Normalization
  elif (module_name == 'lstmLN'):
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  return rnn_cell


# def random_generator (batch_size, z_dim, T_mb, max_seq_len):
#   """Random vector generation.
  
#   Args:
#     - batch_size: size of the random vector
#     - z_dim: dimension of random vector
#     - T_mb: time information for the random vector
#     - max_seq_len: maximum sequence length
    
#   Returns:
#     - Z_mb: generated random vector
#   """
#   Z_mb = list()
#   for i in range(batch_size):
#     temp = np.zeros([max_seq_len, z_dim])
#     temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
#     temp[:T_mb[i],:] = temp_Z
#     Z_mb.append(temp_Z)
#   return Z_mb


# def random_generator(batch_size, z_dim, T_mb, max_seq_len):
#     """
#     Generate a batch of random vectors for the generator input.
    
#     Returns:
#         Z_mb: NumPy array of shape (batch_size, max_seq_len, z_dim)
#     """
#     Z_mb = np.zeros((batch_size, max_seq_len, z_dim), dtype=np.float32)
    
#     for i in range(batch_size):
#         # Fill only up to the actual sequence length
#         Z_mb[i, :T_mb[i], :] = np.random.uniform(0., 1., (T_mb[i], z_dim))
    
#     return Z_mb


def random_generator_tf(batch_size, z_dim, T_mb, max_seq_len):
    """
    Generate a batch of random vectors on GPU for the generator input.
    
    Args:
        batch_size: number of sequences
        z_dim: latent dimension
        T_mb: sequence lengths, shape (batch_size,)
        max_seq_len: maximum sequence length in the batch
    
    Returns:
        Z_mb: tf.Tensor of shape (batch_size, max_seq_len, z_dim)
    """
    # Generate uniform random numbers for all positions
    Z_mb = tf.random.uniform((batch_size, max_seq_len, z_dim), 0.0, 1.0, dtype=tf.float32)

    # Create a mask to zero out positions beyond T_mb
    mask = tf.sequence_mask(T_mb, maxlen=max_seq_len, dtype=tf.float32)  # shape (batch_size, max_seq_len)
    mask = tf.expand_dims(mask, axis=-1)  # shape (batch_size, max_seq_len, 1)

    print(f'Tmb {T_mb.shape}')
    print(f'ZMb {Z_mb.shape}')
    print(f'mask {mask.shape}')
    Z_mb = Z_mb * mask  # zero out beyond sequence length
    return Z_mb


# def batch_generator(data, time, batch_size):
#   """Mini-batch generator.
  
#   Args:
#     - data: time-series data
#     - time: time information
#     - batch_size: the number of samples in each batch
    
#   Returns:
#     - X_mb: time-series data in each batch
#     - T_mb: time information in each batch
#   """
#   no = len(data)
#   idx = np.random.permutation(no)
#   train_idx = idx[:batch_size]     
            
#   X_mb = list(data[i] for i in train_idx)
#   T_mb = list(time[i] for i in train_idx)
  
#   return X_mb, T_mb

from tensorflow.keras.preprocessing.sequence import pad_sequences

# def batch_generator(data, time, batch_size):
#     """
#     Mini-batch generator that returns 3D NumPy arrays.
#     """
#     no = len(data)
#     idx = np.random.permutation(no)
#     train_idx = idx[:batch_size]

#     # Get the selected sequences
#     X_mb = [data[i] for i in train_idx]
#     T_mb = [time[i] for i in train_idx]

#     # Pad sequences to max length in the batch
#     max_len = max(T_mb)
#     X_mb_padded = pad_sequences(
#         X_mb, maxlen=max_len, dtype='float32', padding='post', truncating='post'
#     )

#     T_mb = np.array(T_mb, dtype=np.int32)
#     return X_mb_padded, T_mb
def make_tf_dataset(data, time, batch_size):
    """
    Returns a tf.data.Dataset that yields padded batches asynchronously,
    skipping the last batch if it is smaller than batch_size.
    """
    no = len(data)

    def gen():
        while True:
            idx = np.random.permutation(no)
            for i in range(0, no, batch_size):
                batch_idx = idx[i:i+batch_size]
                
                # Skip batch if it's smaller than batch_size
                if len(batch_idx) < batch_size:
                    continue

                X_mb = [data[j] for j in batch_idx]
                T_mb = [time[j] for j in batch_idx]
                max_len = max(T_mb)
                X_mb_padded = pad_sequences(
                    X_mb, maxlen=max_len, dtype='float32', padding='post', truncating='post'
                )
                yield X_mb_padded, np.array(T_mb, dtype=np.int32)

    output_shape = (None, None, data[0].shape[1])
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    # Shuffle and prefetch for asynchronous GPU usage
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
