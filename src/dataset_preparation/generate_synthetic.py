import numpy as np

# 1. TimeGAN model
from src.dataset_preparation.timegan import timegan
# 2. Data loading
from TimeGAN.data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from TimeGAN.metrics.discriminative_metrics import discriminative_score_metrics
from TimeGAN.metrics.predictive_metrics import predictive_score_metrics
from TimeGAN.metrics.visualization_metrics import visualization
import os
## Data loading
data_name = 'stock'
seq_len = 24

"""
Fineuner for lagllama on given dataset

Args
----
    dataset
        dataset to finetune lagllama on, of class BaseFinetuningDataset
    save_dir
        where to save the model checkpoints

Returns
-------
    Fineunes a lagllama model on all available target columns in the original dataset, saves each to a checkpoint 
    which can be loaded for evaluation

To run, cd into root repo and run
export PYTHONPATH=$(pwd)
poetry run python src/dataset_preparation/generate_synthetic.py 
"""


if data_name in ['stock', 'energy']:
  ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
  # Set number of samples and its dimensions
  no, dim = 10000, 5
  ori_data = sine_data_generation(no, seq_len, dim)
else:
  raise ValueError
    
print(data_name + ' dataset is ready.')

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
# parameters['iterations'] = 10000
parameters['iterations'] = 2000
parameters['batch_size'] = 128

# Run TimeGAN
generated_data = timegan(ori_data, parameters)   
print('Finish Synthetic Data Generation')

metric_iteration = 5
save_dir = f"src/database/synthetic_data/{data_name}"
import pickle
os.makedirs(save_dir, exist_ok=True)
with open(f"{save_dir}/generated_data.pkl", "wb") as f:
    pickle.dump(generated_data, f)

# discriminative_score = list()
# for _ in range(metric_iteration):
#   temp_disc = discriminative_score_metrics(ori_data, generated_data)
#   discriminative_score.append(temp_disc)

# print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)))

# predictive_score = list()
# for tt in range(metric_iteration):
#   temp_pred = predictive_score_metrics(ori_data, generated_data)
#   predictive_score.append(temp_pred)   
    
# print('Predictive score: ' + str(np.round(np.mean(predictive_score), 4)))

# Testing
fig = visualization(ori_data, generated_data, 'pca')
fig.savefig(f"{save_dir}/test_pca.png", dpi=300)  # Save as PNG, 300 DPI
fig = visualization(ori_data, generated_data, 'tsne')
fig.savefig(f"{save_dir}/test_tsne.png", dpi=300)  # Save as PNG, 300 DPI
