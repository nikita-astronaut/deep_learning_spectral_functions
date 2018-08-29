from models import dense_model
from generators import data_generator, uniform_data_generator
from kernels import exponential_kernel
import numpy as np


model = dense_model
epochs = 1000
batch_size = 512
best_weights_path = 'weights/best_weights.hdf5'
best_weights_checkpoint = 'weights/best_weights_checkpoint.hdf5'
best_model_path = 'models/best_model.json'
random_seed = 42
tta_steps = 10
omegas = np.linspace(0, 10.0, 1000)
taus = np.linspace(0, 1.0,  64)
spectral_generator = uniform_data_generator
kernel = exponential_kernel
generator = data_generator