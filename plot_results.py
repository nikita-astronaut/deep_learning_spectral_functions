import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import params

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from generators import data_generator
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
    test_probas = model.predict([corr], batch_size=batch_size, verbose=verbose)
    predictions[0] = test_probas.reshape(test_probas.shape[0])

    for i in range(1, tta_steps):
        test_probas = model.predict([corr], batch_size=batch_size, verbose=verbose)
        predictions[i] = test_probas.reshape(test_probas.shape[0])

    return predictions

nn = model(input_length=len(taus), output_length=len(omegas))
nn.load_weights(filepath=best_weights_path)

trial_spectral_function = spectral_generator(
