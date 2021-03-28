import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard
from deep_bomber.agent_network import network


def main(
        name=f'Koetherminator-{int(time.time())}',
        batch_size=128,
        initial_epoch=0,
        lr=1,
        n_actions=6,
        input_dims=(17, 17, 1),
        n_epochs=120,
        X=None,
        y=None
):

    # Create save dir
    os.makedirs('pre_training', exist_ok=True)

    # Load data
    assert(backend.image_data_format()=='channels_last')
    X = X if X is not None else np.load('deep_bomber/observations.npy')
    y = y if y is not None else np.load('deep_bomber/qs.npy')
    if np.any(y == float('-inf')):
        y = np.where(y == float('-inf'), 0, y)
    # X=np.random.random((10000, *input_dims))
    # y=np.random.random((10000, N_ACTIONS))

    # Train model
    model = network(lr, n_actions, input_dims)
    if initial_epoch != 0:
        model.load_weights('pre_training/latest-network.hdf5')
        print('loaded weights successfully')

    mc = ModelCheckpoint('pre_training/latest-network.hdf5',
                         verbose=1, save_best_only=False,
                         save_weights_only=True, mode='auto', save_freq='epoch')
    mcb = ModelCheckpoint('pre_training/best-network.hdf5',
                          monitor='val_loss', verbose=1, save_best_only=True,
                          save_weights_only=True, mode='auto', save_freq='epoch')

    # Schedule of DenseNet/ResNext
    def schedule(epoch, current_lr):
        if epoch == 30 or epoch == 60 or epoch == 90:
            return current_lr/10
        return current_lr
    lrs = LearningRateScheduler(schedule, verbose=1)
    log = CSVLogger('pre_training/pre_training.log')
    tb = TensorBoard(log_dir=f'pre_training/logs/{name}')

    model.fit(X, y, batch_size=batch_size, epochs=n_epochs,
              callbacks=[mc, mcb, lrs, log, tb], validation_split=0.3,
              shuffle=True, initial_epoch=initial_epoch, verbose=2)


if __name__ == '__main__':
    main(  # Values
        name=f'Koetherminator-{int(time.time())}',
        batch_size=128,
        initial_epoch=0,
        lr=0.1,
        n_actions=6,
        input_dims=(17, 17, 1),
        n_epochs=120
    )

