from time import time
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
    def __init__(self, num_classes, model_name, bit_array_size, wandb_config):
        super(BlasterLSTM, self).__init__()
        # /Users/mobara/Documents/unr/pg2023/projektDL/Blaster/.venv/lib/python3.11/site-packages/torch/nn/modules/rnn.py:82: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
        self.model_name = model_name
        self.flatten = nn.Flatten()
        self.lstm1 = nn.LSTM(
            bit_array_size,
            wandb_config["a_size"],
            batch_first=True,
            dropout=0.0,
        )
        self.linear1 = nn.Linear(wandb_config["a_size"], wandb_config["b_size"])
        self.lstm2 = nn.LSTM(
            wandb_config["b_size"],
            wandb_config["c_size"],
            batch_first=True,
            num_layers=2,
            dropout=wandb_config["dropout"],
        )
        self.linear2 = nn.Linear(wandb_config["c_size"], num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.linear1(x)
        x, _ = self.lstm2(x)
        x = self.linear2(x[:, -1, :])
        return x

    def get_model_name(self):
        return f"./models/BlasterLSTM_{self.model_name}_{int(time())}.pth"
