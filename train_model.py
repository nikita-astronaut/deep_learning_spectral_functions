import numpy as np
import keras
import params
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

model = params.model
epochs = params.max_epochs
batch_size = params.batch_size
best_weights_path = params.best_weights_path
best_weights_checkpoint = params.best_weights_checkpoint
best_model_path = params.best_model_path
random_seed = params.seed
num_folds = params.num_folds
tta_steps = params.tta_steps
omegas = params.omegas
taus = params.taus
spectral_generator = params.spectral_generator
kernel = params.kernel

def get_best_history(history, monitor='val_loss', mode='min'):
    best_iteration = np.argmax(history[monitor]) if mode == 'max' else np.argmin(history[monitor])
    loss = history['loss'][best_iteration]
    acc = history['acc'][best_iteration]
    val_loss = history['val_loss'][best_iteration]
    val_acc = history['val_acc'][best_iteration]

    return best_iteration + 1, loss, acc, val_loss, val_acc

def predict_with_tta(model, corr, verbose=0):
    predictions = np.zeros((tta_steps, len(omegas)))
    test_probas = model.predict([corr], batch_size=batch_size, verbose=verbose)
    predictions[0] = test_probas.reshape(test_probas.shape[0])

    for i in range(1, tta_steps):
        test_probas = model.predict([corr], batch_size=batch_size, verbose=verbose)
        predictions[i] = test_probas.reshape(test_probas.shape[0])

    return predictions

def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=40, verbose=1, min_delta=1e-4, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.1,
            verbose=1, epsilon=1e-4, mode='min'),
        ModelCheckpoint(monitor='val_loss', filepath=best_weights_checkpoint,
            save_best_only=True, save_weights_only=True, mode='min')
    ]

def train_and_evaluate_model(model):
    hist = model.fit_generator(
        data_generator(spectral_generator, kernel, omegas, taus, batch_size),
        steps_per_epoch=100,
        epochs=epochs,
        verbose=2,
        validation_data=data_generator(spectral_generator, kernel, omegas, taus, batch_size),
        validation_steps=10,
        callbacks=get_callbacks()
    )

    best_epoch, loss, acc, val_loss, val_acc = get_best_history(hist.history, monitor='val_loss', mode='min')
    print ()
    print ("Best epoch: {}".format(best_epoch))
    print ("loss: {:0.6f} - val_loss: {:0.6f}}".format(loss, val_loss))
    print ()
    return val_loss

nn = model(input_length=len(taus), output_length=len(omegas))
val_loss = train_and_evaluate_model(nn)
nn.load_weights(filepath=best_weights_checkpoint)
nn.save_weights(filepath=best_weights_path)

