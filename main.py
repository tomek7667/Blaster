import torch
from random import random
from load_data import *

max_chunked_sequence_length = 8
split_coeff = 0.8
print('loading data...')
d = load_data()
print('data loaded')


def prepare_learn_and_test_set():
    learn_data = []
    test_data = []
    for record in d:
        class_name = record['c']
        sequence = record['s']
        for i in range(0, len(sequence), max_chunked_sequence_length):
            chunked_sequence = sequence[i:i+max_chunked_sequence_length]
            if len(chunked_sequence) < max_chunked_sequence_length:
                break
            if random() < split_coeff:
                learn_data.append((class_name, chunked_sequence))
            else:
                test_data.append((class_name, chunked_sequence))

    return learn_data, test_data

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
    return

if __name__ == "__main__":
    main()
    exit(0)
