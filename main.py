from torch import nn, optim
from random import random
from load_data import *
from model import *
from dataset import *

max_chunked_sequence_length = 8
split_coeff = 0.8
print('loading data.sequence_array ..')
d = load_data("prepared/prepared_1697562094237-short.json")
print('data loaded')


def prepare_learn_and_test_set():
    learn_data = []
    test_data = []
    for record in d:
        class_name = record['c']
        sequence = record['s']
        sequence_array = []
        for i in range(0, len(sequence), max_chunked_sequence_length):
            chunked_sequence = sequence[i:i+max_chunked_sequence_length]
            if len(chunked_sequence) < max_chunked_sequence_length:
                break
            sequence_array.append(chunked_sequence)
        if random() < split_coeff:
            learn_data.append((class_name, sequence_array))
        else:
            test_data.append((class_name, sequence_array))

    return learn_data, test_data

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def main():
    print('preparing learn and test set...')
    learn_data, test_data = prepare_learn_and_test_set()
    print('learn and test set prepared')
    learn_data_length = len(learn_data)
    test_data_length = len(test_data)
    total_data_length = learn_data_length + test_data_length
    print(f"Learn data length: {learn_data_length}")
    print(f"Test data length: {test_data_length}")
    print(f"split coeff = {split_coeff}, {learn_data_length/total_data_length*100}%")
    device = get_device()
    train_dataset = BacteriaDataset(learn_data)
    test_dataset = BacteriaDataset(test_data)
    print(train_dataset)
    print(test_dataset)
    train_dataset[0]
    train_dataset[1]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
 
    input_size = train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2]
    num_classes = len(train_dataset.classes)
    print(f"input_size = {input_size}")
    print(f"num_classes = {num_classes}")
    model = Blaster(input_size, num_classes).to(device)
    print(model)
    # total = 0
    # for name, param in model.named_parameters():
    # 	# flatten was skipped in named parameters and other layers
    # 	# because they dont have any named params. in fact they dont have any params at all.
    # 	# Flatten layer just reorders the tensor, so it doesnt have any params.
    # 	print(f"Layer: {name} | Size: {param.size()} | Num el: {param.numel()}")
    # 	total += param.numel()

    # print(f"{total=}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
 

    

if __name__ == "__main__":
    main()
    exit(0)