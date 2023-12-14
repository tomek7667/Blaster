from torch import nn, optim, stack, no_grad, tensor
from torchmetrics import ConfusionMatrix
from sklearn.metrics import f1_score
from random import random, shuffle
from load_data import *
from models import *
from dataset import *
import matplotlib.pyplot as plt
import wandb
import time

VERSION = "1.0.0"
split_coeff = 0.8
EPOCHS = 11
LEARNING_RATE = 0.0001
MAX_SEQUENCE_LENGTH = 700  # 30_000
BATCH_SIZE = 5
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
            time_start = time.time()
            for inputs, targets in train_loader:
                i += 1
                if i % 100 == 0:
                    took = time.time() - time_start
                    print(
                        f"Progress: {i}/{train_loader_length} = {100*i/train_loader_length:2f}%, Epoch: {epoch+1}/{num_epochs}, took: {took:.2f}s"
                    )
                    time_start = time.time()
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
    print(f"{average_loss = }")

def evaluate_model(model, data_loader, model_name, device):
    all_predictions = []
    all_labels =[]
    model.eval()
    with no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    score = f1_score(all_labels, all_predictions, average="micro")
    print(f"{score = }")
    wandb.log(
        {
            "f1_score": score,
            "model": model_name
        }
    )

def ts():
    return int(time.time())

def generate_confusion_matrix(model, data_loader, model_name, device):
    all_predictions = []
    all_labels =[]
    model.eval()
    with no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted)
            all_labels.extend(labels)
    # open(f"debug/{ts()}_all_predictions_{model_name}.txt", "w").write(f"{all_predictions}")
    # open(f"debug/{ts()}__all_labels_{model_name}.txt", "w").write(f"{all_labels}")
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(data_loader.dataset.classes))
    cm = confmat(tensor(all_predictions), tensor(all_labels))
    p = f"results/{ts()}_confusion_matrix_{model_name}.txt"
    print(f"Saving to {p}")
    open(p, "w").write(str(cm))
    return cm

def visualize_confusion_matrix(cm, classes, model_name):
    figsize = (len(classes), len(classes))
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j].item()
            ax.text(
                x=j,
                y=i,
                s=value,
                va="center",
                ha="center",
                size="xx-large",
                color="white",
            )
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    p = f"results/{ts()}_confusion_matrix_{model_name}.png"
    print(f"Saving to {p}")
    plt.savefig(p)
    # plt.show()

def save_model_params(model_name, classes, sequence_length, num_classes, wandb_config):
    open(
            f"models/{model_name}.py",
            "w",
        ).write( # c_size = {wandb.config.c_size}
            f"""
classes = {classes}
sequence_length = {sequence_length}
num_classes = {num_classes}
dropout = {wandb_config.dropout}
a_size = {wandb_config.a_size}
b_size = {wandb_config.b_size}
batch_size = {wandb_config.batch_size}
optimizer = "{wandb_config.optimizer}"
learning_rate = {wandb_config.learning_rate}"""
        )

def main():
    sweep_config = {
        "method": "random",
        "parameters": {
            "dropout": {"values": [0.2]},
            "optimizer": {"values": ["adam"]},
            "learning_rate": {"values": [LEARNING_RATE, LEARNING_RATE * 10, int(LEARNING_RATE / 10)]},
            "batch_size": {"values": [BATCH_SIZE, BATCH_SIZE * 2, int(BATCH_SIZE / 2)]},
            "a_size": {"values": [64, 192, 308]},
            "b_size": {"values": [16, 48, 128]},
            # "c_size": {"values": [128, 256, 512]},
        },
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=start, count=99)


def start():
    try:
        wandb.init(project="Blaster-test", tags=[f"{VERSION=}"])
        print(
            f"Starting sweep with {wandb.config.dropout=}, {wandb.config.optimizer=}, {wandb.config.learning_rate=}, {wandb.config.batch_size=}, {wandb.config.a_size=}, {wandb.config.b_size=}"
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
        print(f"{device = }")
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
        save_model_params(model_name, classes, sequence_length, num_classes, wandb.config)
        model = BlasterLSTM(num_classes, model_name, 4, wandb.config).to(device)
        total_parameters_number = 0
        for name, param in model.named_parameters():
            # flatten was skipped in named parameters and other layers
            # because they dont have any named params. in fact they dont have any params at all.
            # Flatten layer just reorders the tensor, so it doesnt have any params.
            print(f"Layer: {name} | Size: {param.size()} | Num el: {param.numel()}")
            total_parameters_number += param.numel()
        print(f"{total_parameters_number = }")

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
        print("Evaluating model...")
        evaluate_model(model, test_loader, model_name, device)
        print("Model evaluated")
        print("Generating confusion matrix...")
        cm = generate_confusion_matrix(model, test_loader, model_name, device)
        print("Confusion matrix generated")
        print("Visualizing confusion matrix...")
        visualize_confusion_matrix(cm, classes, model_name)
        wandb.finish()
    except Exception as e:
        print(e)
        wandb.finish()
        raise e


if __name__ == "__main__":
    main()
    exit(0)
