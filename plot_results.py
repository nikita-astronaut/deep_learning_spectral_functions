import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import params
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from generators import data_generator, correlator_generator
from keras import backend as K

model = params.model
epochs = params.epochs
batch_size = params.batch_size
best_weights_path = params.best_weights_path
best_weights_checkpoint = params.best_weights_checkpoint
best_model_path = params.best_model_path
random_seed = params.random_seed
# num_folds = params.num_folds
tta_steps = params.tta_steps
omegas = params.omegas
taus = params.taus
spectral_generator = params.spectral_generator
kernel = params.kernel

def predict_with_tta(model, corr, verbose=0):
    predictions = np.zeros((tta_steps, len(omegas)))
    test_probas = model.predict(corr[np.newaxis, :], batch_size=batch_size, verbose=verbose)
    predictions[0] = test_probas[0]

    for i in range(1, tta_steps):
        test_probas = model.predict(corr[np.newaxis, :], batch_size=batch_size, verbose=verbose)
        predictions[i] = test_probas[0]
    return predictions.mean(axis = 0), predictions.std(axis = 0)

nn = model(input_length=len(taus), output_length=len(omegas))
nn.load_weights(filepath=best_weights_checkpoint)

trial_spectral_function = np.zeros(len(omegas))
trial_spectral_function[16] = 1.0
trial_correlator = correlator_generator(trial_spectral_function, kernel, omegas, taus)
mean, std = predict_with_tta(nn, trial_correlator)
print(omegas, mean, std)

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.errorbar(omegas, mean, std, markersize = 2, fmt = '^', color = 'red', elinewidth=0.3, label='reconstructed')
plt.scatter(omegas, trial_spectral_function, color='blue', label='true')
plt.legend(loc='upper left', ncol=2, fontsize=10)
plt.xlabel('$\\omega$', fontsize = 14)
plt.ylabel('$\\rho(\\omega)$', fontsize = 14)
plt.grid(True, alpha = 0.5, linestyle='--')
plt.savefig('./plots/delta_exponential.pdf')
