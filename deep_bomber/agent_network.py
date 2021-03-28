from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

def network(lr, n_actions, input_dims):
    inputs = Input(shape=input_dims)
    x = Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(n_actions, activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=lr), 
                  loss='mean_squared_error')
    return model