cartpole = {'network': 'mlp',
            'lr': 1e-4,
            'total_timesteps': 10000000,
            'buffer_size': 50000,
            'exploration_fraction': 0.001,
            'exploration_final_eps': 0.02,
            'print_freq': 10,
            'train_freq': 4,
            'target_network_update_freq': 1000,
            'gamma': 0.99
}

pong = {
        'network': "conv_only",
        'convs': [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        'hiddens': [256],
        'dueling': True,
        'lr': 1e-4,
        'total_timesteps': int(1e7),
        'buffer_size': 10000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
        'print_freq': 1,
        'train_freq': 4,
        'learning_starts': 10000,
        'target_network_update_freq': 1000,
        'gamma': 0.99
}