from torch import nn, optim
from random import random, shuffle
from load_data import *
from models import *
from dataset import *
import wandb

split_coeff = 0.8
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_SEQUENCE_LENGTH = 120_000
BATCH_SIZE = 32
path = "prepared/prepared_1697562094237-short.json"
CHECKPOINT_MODEL_PATH = "models/Blaster_B2dd6b9_1699353998.pth"

from models.B2dd6b9 import *


wandb.init(
    project="Blaster",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "CNN",
        "dataset": "BacteriaDataset",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
)


def random_name():
    return model_name


def prepare_learn_and_test_set(database):
    learn_data = []
    test_data = []

    dict_classes = {}
    longest_sequence = 0
    for record in database:
        class_name = record["c"]
        sequence = record["s"]

        if not class_name in dict_classes:
            dict_classes[class_name] = []
        dict_classes[class_name].append(sequence)
        if len(sequence) > longest_sequence:
            longest_sequence = len(sequence)
    if longest_sequence > MAX_SEQUENCE_LENGTH:
        longest_sequence = MAX_SEQUENCE_LENGTH
    print(
        f"Longest sequence length: {longest_sequence} - to that number other sequences are going to be prefixidly padded"
    )
    for class_name, sequences in dict_classes.items():
        print(
            f"Dividing class {class_name} into learn and test set... (total length={len(sequences)})"
        )
        for sequence in sequences:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
            # zero pad the sequence with '-' characters.
            padded_sequence = "-" * (longest_sequence - len(sequence)) + sequence
            sequence_array = []
            for i in range(0, len(padded_sequence), max_chunked_sequence_length):
                chunked_sequence = padded_sequence[i : i + max_chunked_sequence_length]
                if len(chunked_sequence) < max_chunked_sequence_length:
                    break
                sequence_array.append(chunked_sequence)
            if random() < split_coeff:
                learn_data.append((class_name, sequence_array))
            else:
                test_data.append((class_name, sequence_array))
    print("Shuffling learn and test set...")
    shuffle(learn_data)
    shuffle(test_data)
    return learn_data, test_data


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    previous_running_loss,
    num_epochs=10,
):
    print(f"Training model {num_epochs=}...")
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = previous_running_loss
            train_loader_length = len(train_loader)
            i = 0
            for inputs, targets in train_loader:
                i += 1
                print(
                    f"Progress: {i}/{train_loader_length} = {100*i/train_loader_length:2f}%, Epoch: {epoch+1}/{num_epochs}"
                )
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
    except Exception as e:
        raise e


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
    wandb.log(
        {
            "average_test_loss": average_loss,
        }
    )
    print(f"Loss: {average_loss}")


def main():
    print("loading data.sequence_array ..")
    database = load_data(path)
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
    print(f"{device=}")
    train_dataset = BacteriaDataset(learn_data)
    test_dataset = BacteriaDataset(test_data)
    print(train_dataset)
    print(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    input_size = len(train_dataset[0][0])
    print(f"input_size = {input_size}")
    num_classes = len(train_dataset.classes)
    print(f"input_size = {input_size}")
    print(f"num_classes = {num_classes}")
    classes = train_dataset.classes
    print(f"classes = {classes}")
    model_name = random_name()
    open(f"models/{model_name}.py", "w+").write(
        f"""
input_size = {input_size}
max_chunked_sequence_length = {max_chunked_sequence_length}
classes = {classes}
model_name = "{model_name}"
"""
    )
    model = BlasterLSTM(
        input_size, max_chunked_sequence_length, len(classes), model_name
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_MODEL_PATH)

    model.load_state_dict(checkpoint["model_state_dict"])
    total = 0
    for name, param in model.named_parameters():
        # flatten was skipped in named parameters and other layers
        # because they dont have any named params. in fact they dont have any params at all.
        # Flatten layer just reorders the tensor, so it doesnt have any params.
        print(f"Layer: {name} | Size: {param.size()} | Num el: {param.numel()}")
        total += param.numel()
    print(f"{total=}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    running_loss = checkpoint["loss"]
    train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        running_loss,
        num_epochs=EPOCHS,
    )
    # save the model
    torch.save(model.state_dict(), model.get_model_name())
    # test the model
    test_model(model, test_loader, criterion, device)
    wandb.finish()


if __name__ == "__main__":
    main()
    exit(0)
