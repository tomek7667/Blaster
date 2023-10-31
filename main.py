from torch import nn, optim
from random import random, shuffle
from load_data import *
from model import *
from dataset import *

max_chunked_sequence_length = 8
split_coeff = 0.8
path = "prepared/prepared_1697562094237-short.json"

def prepare_learn_and_test_set(database):
    learn_data = []
    test_data = []

    dict_classes = {}
    longest_sequence = 0
    for record in database:
        class_name = record['c']
        sequence = record['s']
        
        if not class_name in dict_classes:
            dict_classes[class_name] = []
        dict_classes[class_name].append(sequence)
        if len(sequence) > longest_sequence:
            longest_sequence = len(sequence)
    print(f'Longest sequence length: {longest_sequence} - to that number other sequences are going to be prefixidly padded')
    for class_name, sequences in dict_classes.items():
        print(f'Dividing class {class_name} into learn and test set... (total length={len(sequences)})')
        for sequence in sequences:
            # zero pad the sequence with '-' characters.
            padded_sequence = '-' * (longest_sequence - len(sequence)) + sequence
            sequence_array = []
            for i in range(0, len(padded_sequence), max_chunked_sequence_length):
                chunked_sequence = padded_sequence[i:i+max_chunked_sequence_length]
                if len(chunked_sequence) < max_chunked_sequence_length:
                    break
                sequence_array.append(chunked_sequence)
            if random() < split_coeff:
                learn_data.append((class_name, sequence_array))
            else:
                test_data.append((class_name, sequence_array))
    print('Shuffling learn and test set...')
    shuffle(learn_data)
    shuffle(test_data)
    return learn_data, test_data

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    print('Training model...')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def main():
    print('loading data.sequence_array ..')
    database = load_data(path)
    print('data loaded')
    print('preparing learn and test set...')
    learn_data, test_data = prepare_learn_and_test_set(database)
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
    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)
    input_size = len(train_dataset[0][0])
    num_classes = len(train_dataset.classes)
    print(f"input_size = {input_size}")
    print(f"num_classes = {num_classes}")
    classes = train_dataset.classes
    print(f"classes = {classes}")
    model = Blaster(input_size, len(classes)).to(device)
    print(model)
    # total = 0
    # for name, param in model.named_parameters():
    #     # flatten was skipped in named parameters and other layers
    #     # because they dont have any named params. in fact they dont have any params at all.
    #     # Flatten layer just reorders the tensor, so it doesnt have any params.
    #     print(f"Layer: {name} | Size: {param.size()} | Num el: {param.numel()}")
    #     total += param.numel()
    # print(f"{total=}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
 

    

if __name__ == "__main__":
    main()
    exit(0)
