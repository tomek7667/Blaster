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


class Blaster(nn.Module):
    def __init__(self, input_size, chunk_size, num_classes, bit_array_size=4):
        super(Blaster, self).__init__()
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
        return f"./models/Blaster_{int(time())}.pth"
