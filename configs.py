from PIL import Image

model_configurations = {
    'resnet50': {
        'feature_num': 2048,
        'feature_map_channels': 2048,
        'policy_conv': False,
        'policy_hidden_dim': 1024,
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224
    }
}


train_configurations = {
    'resnet': {
        'backbone_lr': 0.01,
        'fc_stage_1_lr': 0.1,
        'fc_stage_3_lr': 0.1,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'Nesterov': True,
        'batch_size': 32,
        'epoch_num': 20,
    }
}
