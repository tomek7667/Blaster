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
        self.model_name = model_name
        
        self.embedding = nn.Linear(bit_array_size, wandb_config["a_size"])
        
        self.lstm1 = nn.LSTM(
            wandb_config["a_size"],
            wandb_config["b_size"],
            batch_first=True,
            dropout=0.0,
        )
        # self.linear1 = nn.Linear(wandb_config["a_size"], wandb_config["b_size"])
        # self.lstm2 = nn.LSTM(
        #     wandb_config["b_size"],
        #     wandb_config["c_size"],
        #     batch_first=True,
        #     num_layers=2,
        #     dropout=wandb_config["dropout"],
        #     bidirectional=True,
        # )
        # self.linear2 = nn.Linear(wandb_config["c_size"] * 2, num_classes)
        self.linear2 = nn.Linear(wandb_config["b_size"], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        # x = self.linear1(x)
        # x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        # x = self.softmax(x)
        # open("./debug/blaster_x1.txt", "w", encoding="utf-8").write(str(x))
        x = self.linear2(x[:, -1, :])
        # open("./debug/blaster_x2.txt", "w", encoding="utf-8").write(str(x))
        # x = self.softmax(x)
        # open("./debug/blaster_x3.txt", "w", encoding="utf-8").write(str(x))
        # exit(0)
        return x

    def get_model_name(self):
        return f"./models/BlasterLSTM_{self.model_name}_{int(time())}.pth"

class ImprovedBlasterLSTM(nn.Module):
    def __init__(self, model_name, num_classes, bit_array_size, wandb_config):
        super(ImprovedBlasterLSTM, self).__init__()
        self.model_name = model_name

        self.embedding = nn.Linear(bit_array_size, wandb_config["a_size"])
        
        self.lstm1 = nn.LSTM(
            wandb_config["a_size"],
            wandb_config["a_size"],
            batch_first=True,
            dropout=wandb_config["dropout"],
            num_layers=2,
        )
        
        self.lstm2 = nn.LSTM(
            wandb_config["a_size"],
            wandb_config["b_size"],
            batch_first=True,
            dropout=wandb_config["dropout"],
            bidirectional=True,
            num_layers=2,
        )

        self.lstm3 = nn.LSTM(
            wandb_config["b_size"] * 2,
            wandb_config["c_size"],
            batch_first=True,
            dropout=wandb_config["dropout"],
            bidirectional=True,
            num_layers=2,
        )

        self.fc = nn.Linear(wandb_config["c_size"] * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.shape)
        x = self.embedding(x)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        
        return x
    
    def get_model_name(self):
        return f"./models/BlasterLSTM_{self.model_name}_{int(time())}.pth"
