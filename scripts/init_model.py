import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, state_size, layers_config, action_size):
        super(CustomModel, self).__init__()

        self.layers = nn.ModuleList()
        prev_channels = state_size

        for config in layers_config:
            neurons, batch_norm, dropout = config

            # Add linear layer
            self.layers.append(nn.Linear(prev_channels, neurons))

            # Add batch normalization layer if specified
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(neurons))

            # Add dropout layer if specified
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

            prev_channels = neurons

        # Output layer
        self.out_layer = nn.Linear(prev_channels, action_size)

        # Metrics utiliy
        self.model_metrics = []

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.out_layer(x)
        return x

    def save(self, model, name, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': self.model_metrics
        }, path+name+'.pth')

    def add_epoch_metrics(self, epoch_metrics):
        self.model_metrics.append(epoch_metrics)

    def get_model_metrics(self):
        return self.model_metrics
