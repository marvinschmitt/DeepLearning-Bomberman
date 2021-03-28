import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard
from adapter.bomberman_adapter import BombermanEnvironment
from deep_bomber.agent_network import network
    
if __name__ == '__main__':
    # Values
    NAME = f'Kötherminator-{int(time.time())}'
    BATCH_SIZE = 128
    INITIAL_EPOCH = 0
    LEARNING_RATE = 0.1
    N_ACTIONS = 6
    input_dims = BombermanEnvironment().observation_shape


    # Create save dir
    os.makedirs('pre_training', exist_ok=True)

    # Load data
    assert(backend.image_data_format()=='channels_last')
    X = np.load('X.npy')
    y = np.load('y.npy')
    #X=np.random.random((10000, *input_dims))
    #y=np.random.random((10000, N_ACTIONS))


    # Train model
    model = network(LEARNING_RATE, N_ACTIONS, input_dims)
    if INITIAL_EPOCH != 0:
        model.load_weights('pre_training/latest-network.hdf5')
        print('loaded weights successfully')

    mc = ModelCheckpoint('pre_training/latest-network.hdf5', 
                         verbose=1, save_best_only=False, 
                         save_weights_only=True, mode='auto', save_freq='epoch')
    mcb = ModelCheckpoint('pre_training/best-network.hdf5', 
                          monitor='val_loss', verbose=1, save_best_only=True,
                          save_weights_only=True, mode='auto', save_freq='epoch')
    #Schedule of DenseNet/ResNext
    def schedule(epoch, current_lr):
      if epoch == 30 or epoch == 60 or epoch == 90:
        return current_lr/10
      return current_lr
    lrs = LearningRateScheduler(schedule, verbose=1)
    log = CSVLogger('pre_training/pre_training.log')
    tb = TensorBoard(log_dir=f'pre_training/logs/{NAME}')

    model.fit(X, y, batch_size=BATCH_SIZE, epochs=120, 
              callbacks = [mc, mcb, lrs, log, tb], validation_split = 0.3, 
              shuffle=True, initial_epoch=INITIAL_EPOCH, verbose=2)