from torch import nn, optim, squeeze
import time
from random import random, shuffle
from load_data import *
from models import *
from dataset import *
import wandb

split_coeff = 0.8
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_SEQUENCE_LENGTH = 700  # 30_000
BATCH_SIZE = 16
MUTATION_IN_SEQUENCE_COEFFICIENT = 0.3
MUTATIONS_PER_SEQUENCE = 5
path = "prepared/prepared_1697562094237-short.json"


def random_name():
    return "B" + "".join([hex(int(random() * 16))[2:] for _ in range(6)])


def prepare_learn_and_test_set(database):
    learn_data = []
    test_data = []
    dict_classes = {}
    longest_sequence_length = 0
    for record in database:
        class_name = record["c"]
        sequence = record["s"]

        if not class_name in dict_classes:
            dict_classes[class_name] = []
        dict_classes[class_name].append(sequence)
        if len(sequence) > longest_sequence_length:
            longest_sequence_length = len(sequence)
    if longest_sequence_length > MAX_SEQUENCE_LENGTH:
        longest_sequence_length = MAX_SEQUENCE_LENGTH
    print(
        f"Longest sequence length: {longest_sequence_length} - to that number other sequences are going to be prefixidly padded"
    )
    for class_name, sequences in dict_classes.items():
        print(
            f"Dividing class {class_name} into learn and test set... (total length={len(sequences)})"
        )
        for sequence in sequences:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
            padded_sequence = "-" * (longest_sequence_length - len(sequence)) + sequence
            if random() < split_coeff:
                learn_data.append((class_name, padded_sequence))
            else:
                test_data.append((class_name, padded_sequence))
    print("Shuffling learn and test set...")
    print(learn_data[0])
    shuffle(learn_data)
    shuffle(test_data)
    return learn_data, test_data


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print("Training model...")
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            train_loader_length = len(train_loader)
            i = 0
            time_start = time()
            for inputs, targets in train_loader:
                i += 1
                if i % 100 == 0:
                    took = time() - time_start
                    print(
                        f"Progress: {i}/{train_loader_length} = {100*i/train_loader_length:2f}%, Epoch: {epoch+1}/{num_epochs}, took: {took:.2f}s"
                    )
                    time_start = time()
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                running_loss += loss_item
                wandb.log(
                    {
                        "loss": loss_item,
                        "epoch": epoch,
                    }
                )
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}"
            )
    except KeyboardInterrupt:
        print("Interrupted - saving model...")
        torch.save(
            {
                "epoch": EPOCHS,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": running_loss / len(train_loader),
            },
            model.get_model_name(),
        )

        print("Model saved")
        wandb.finish()
        exit(0)


def test_model(model, test_loader, criterion, device):
    print("Testing model with F1 score...")
    model.eval()
    running_loss = 0.0
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_item = loss.item()
        running_loss += loss_item
        print(f"Loss: {loss.item()}")
        wandb.log(
            {
                "test_loss": loss_item,
            }
        )
    average_loss = running_loss / len(test_loader)
    print(f"{average_loss =}")


def main():
    sweep_config = {
        "method": "random",
        "parameters": {
            "dropout": {"values": [0.2, 0.4]},
            "optimizer": {"values": ["adam"]},
            "learning_rate": {"values": [0.01]},
            "batch_size": {"values": [3, 12]},
            "a_size": {"values": [128, 256]},
            "b_size": {"values": [16, 48, 128]},
            "c_size": {"values": [128, 256, 512]},
        },
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=start, count=15)


def start():
    wandb.init()
    print(
        f"Starting sweep with {wandb.config.dropout=}, {wandb.config.optimizer=}, {wandb.config.learning_rate=}, {wandb.config.batch_size=}, {wandb.config.a_size=}, {wandb.config.b_size=}, {wandb.config.c_size=}"
    )
    print("loading data.sequence_array ..")
    database = load_data(
        path,
        MUTATIONS_PER_SEQUENCE,
        int(MAX_SEQUENCE_LENGTH * MUTATION_IN_SEQUENCE_COEFFICIENT),
    )
    print("data loaded")
    print("preparing learn and test set...")
    learn_data, test_data = prepare_learn_and_test_set(database)
    print("learn and test set prepared")
    learn_data_length = len(learn_data)
    test_data_length = len(test_data)
    total_data_length = learn_data_length + test_data_length
    print(f"Learn data length: {learn_data_length}")
    print(f"Test data length: {test_data_length}")
    print(f"split coeff = {split_coeff}, {learn_data_length/total_data_length*100}%")
    device = get_device()
    print(f"{device =}")
    train_dataset = BacteriaDataset(learn_data)
    test_dataset = BacteriaDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size)

    sequence_length = len(train_dataset[0][0])
    print(f"{sequence_length=}")

    num_classes = len(train_dataset.classes)
    print(f"{num_classes=}")

    classes = train_dataset.classes
    model_name = random_name()
    open(
        f"models/{model_name}.py",
        "w",
    ).write(
        f"""
classes = {classes}
sequence_length = {sequence_length}
num_classes = {num_classes}
dropout = {wandb.config.dropout}
a_size = {wandb.config.a_size}
b_size = {wandb.config.b_size}
c_size = {wandb.config.c_size}
batch_size = {wandb.config.batch_size}
optimizer = "{wandb.config.optimizer}"
learning_rate = {wandb.config.learning_rate}           		
"""
    )
    model = BlasterLSTM(num_classes, model_name, 4, wandb.config).to(device)
    total_parameters_number = 0
    for name, param in model.named_parameters():
        # flatten was skipped in named parameters and other layers
        # because they dont have any named params. in fact they dont have any params at all.
        # Flatten layer just reorders the tensor, so it doesnt have any params.
        print(f"Layer: {name} | Size: {param.size()} | Num el: {param.numel()}")
        total_parameters_number += param.numel()
    print(f"{total_parameters_number =}")

    criterion = nn.CrossEntropyLoss()

    optimizer_type = wandb.config.optimizer
    optimizer = None
    if optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=wandb.config.learning_rate, momentum=0.9
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=EPOCHS)
    torch.save(model.state_dict(), model.get_model_name())
    print("Training finished")
    print("Starting testing...")
    test_model(model, test_loader, criterion, device)
    print("Testing finished")
    wandb.finish()


if __name__ == "__main__":
    main()
    exit(0)
