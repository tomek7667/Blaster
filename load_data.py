import json
from random import random, randint, choice

available_nucleotides = [
    "G",
    "A",
    "T",
    "C",
    "N",
    "R",
    "Y",
    "K",
    "M",
    "S",
    "W",
    "B",
    "D",
    "H",
    "V",
]


def apply_single_mutation(sequence):
    # depending on the random number, we will either insert, delete or replace a nucleotide
    random_number = random()

    if random_number < 0.33:
        # insert a nucleotide
        random_index = randint(0, len(sequence))
        random_nucleotide = choice(available_nucleotides)
        sequence = sequence[:random_index] + random_nucleotide + sequence[random_index:]
    elif random_number < 0.66:
        # delete a nucleotide
        random_index = randint(0, len(sequence) - 1)
        sequence = sequence[:random_index] + sequence[random_index + 1 :]
    else:
        # replace a nucleotide
        random_index = randint(0, len(sequence) - 1)
        random_nucleotide = choice(available_nucleotides)
        sequence = (
            sequence[:random_index] + random_nucleotide + sequence[random_index + 1 :]
        )


# mutations_multiplier is the number of mutated sequences that will be generated for each sequence
# mutations_degree is the number of mutations that will be applied to each sequence
def load_data(
    path="prepared/prepared_1697562094237-short.json",
    mutations_multiplier=4,
    mutations_degree=50,
):
    with open(path, "r") as f:
        final_arr = []
        print(f"Loading data from {path}...")
        sequence_objects = json.load(f)
        for sequence_object in sequence_objects:
            final_arr.append(sequence_object)
            class_name = sequence_object["c"]
            sequence = sequence_object["s"]
            for _ in range(mutations_multiplier):
                mutated_sequence = sequence
                for _ in range(mutations_degree):
                    apply_single_mutation(mutated_sequence)
                final_arr.append({"c": class_name, "s": mutated_sequence})
        print(
            f"Loaded and generated in total of {len(final_arr)} sequences. (from {len(sequence_objects)} original sequences)"
        )
        return final_arr


if __name__ == "__main__":
    load_data()
