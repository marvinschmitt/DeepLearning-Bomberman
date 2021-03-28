from deep_bomber.agent_network import network

if __name__ == '__main__':
    lr = 1
    n_actions = 6
    input_dims = (17, 17, 1)

    model = network(lr, n_actions, input_dims)

    model.load_weights('../pre_training/best-network.hdf5')
    model.save("../pre_training/best-network.hdf5")

