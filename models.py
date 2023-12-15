from time import time
import torch
from torch import nn, cuda, backends


def get_device():
    device = (
        "cuda"
        if cuda.is_available()
        else "mps"
        if backends.mps.is_available()
        else "cpu"
    )
    return device


class BlasterMultilayerPerceptron(nn.Module):
    @staticmethod
    def prefix():
        return "BlasterMLPv1"

    def __init__(
        self, input_size, chunk_size, num_classes, model_name, bit_array_size=4
    ):
        super(BlasterMultilayerPerceptron, self).__init__()
        self.model_name = model_name
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size * chunk_size * bit_array_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        # This will give the x a length of input_size * chunk_size * 4 (bit-array one-hot)
        logits = self.linear_relu_stack(x)
        return logits

    def get_model_name(self):
        return (
            f"./models/BlasterMultilayerPerceptron_{self.model_name}_{int(time())}.pth"
        )


class BlasterLSTM(nn.Module):
    @staticmethod
    def prefix():
        return "BlasterLSTMv2"
    
    def __init__(self, num_classes, model_name, bit_array_size, wandb_config):
        super(BlasterLSTM, self).__init__()
        self.model_name = model_name
        
        self.embedding = nn.Linear(bit_array_size, wandb_config["a_size"])
        
        self.lstm1 = nn.LSTM(
            wandb_config["a_size"],
            wandb_config["b_size"],
            batch_first=True,
            dropout=0.0,
        )
        self.linear1 = nn.Linear(wandb_config["b_size"], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.linear1(x[:, -1, :])
        return x

    def get_model_name(self):
        return f"./models/BlasterLSTM_{self.model_name}_{int(time())}.pth"

class BlasterLSTMv3(nn.Module):
    @staticmethod
    def prefix():
        return "BlasterLSTMv3"
    
    def __init__(self, num_classes: int, model_name: str, bit_array_size: int, wandb_config, sequence_length: int):
        super(BlasterLSTMv3, self).__init__()
        self.model_name = model_name
        self.should_use_softmax = wandb_config["should_use_softmax"]
        embedding_out_size = int((sequence_length * bit_array_size) / wandb_config["first_layer_chunk"])
        self.embedding = nn.Linear(bit_array_size * sequence_length, embedding_out_size)
        
        self.lstm1 = nn.LSTM(
            embedding_out_size,
            wandb_config["b_size"],
            batch_first=True,
            num_layers=wandb_config["lstm_layers"],
            bidirectional=wandb_config["is_bidirectional"],
            dropout=wandb_config["dropout"],
        )
        linear1_input_size = wandb_config["b_size"] * 2 if wandb_config["is_bidirectional"] else wandb_config["b_size"]
        
        self.linear1 = nn.Linear(linear1_input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 1 sequence: [[0,0,0,1], [0,1,0,0], ... [ pseudo-one-hot ]]
        # Flatten the sequence: [0,0,0,1,0,1,0,0, ... , pseudo-one-hot]
        x = x.view(x.shape[0], -1)
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.linear1(x)
        if self.should_use_softmax:
            x = self.softmax(x)
        return x

    def get_model_name(self):
        return f"./models/BlasterLSTM_{self.model_name}_{int(time())}.pth"

    def get_model_name(self):
        return f"./models/BlasterLSTM_{self.model_name}_{int(time())}.pth"

