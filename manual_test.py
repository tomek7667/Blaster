from torch import load, from_numpy
import numpy as np
from model import Blaster, get_device
from models.Bba162b import *

# labels = {0: "Cyanobacteria",
#           1: "Proteobacteria",
#           2: "Firmicutes",
#           3: "Spirochaetes",
#           4: "Actinobacteria",
#           5: "Tenericutes",
#           6: "Pseudomonadota",
#           7: "Deinococcus-Thermus"}
classes = [
    "Fusobacteriota",
    "Fibrobacteres",
    "Cyanobacteria",
    "Bacteroidota",
    "Mycoplasmatota",
    "Aquificota",
    "Verrucomicrobia",
    "Fusobacteria",
    "Proteobacteria",
    "Chloroflexi",
    "Thermotogae",
    "Deinococcus-Thermus",
    "Pseudomonadota",
    "Myxococcota",
    "Actinomycetota",
    "Planctomycetota",
    "Actinobacteria",
    "Chlamydiae",
    "Bacillota",
    "Spirochaetes",
    "Campylobacterota",
    "Firmicutes",
    "Chlorobi",
    "Tenericutes",
]


def index_to_class_result_mapper(values):
    result = []
    print(values[0][0])
    print(values[0][1])
    print(len(values))
    # l = values.numpy()
    print("a:")
    print(values[0].size())
    print(values[0].size()[0])
    for i in range(values[0].size()[0]):
        value = values[0][i].item()
        result.append({"className": classes[i], "certainty": value})
    return result


def sequence_to_model(sequence: str):
    sequence = sequence[: input_size * 8]
    # zero pad the sequence with '-' characters.
    padded_sequence = "-" * (input_size * 8 - len(sequence)) + sequence
    sequence_array = []
    for i in range(0, len(padded_sequence), max_chunked_sequence_length):
        chunked_sequence = padded_sequence[i : i + max_chunked_sequence_length]
        if len(chunked_sequence) < max_chunked_sequence_length:
            break
        sequence_array.append(chunked_sequence)
    return from_numpy(np.array([sequence_to_one_hot(sequence_array)])).float()


def sequence_to_one_hot(sequence):
    nucleotide_to_index = {"G": 0, "A": 1, "T": 2, "C": 3}
    one_hot_sequence = np.zeros(
        (len(sequence), len(sequence[0]), len(nucleotide_to_index))
    )
    for i, seq in enumerate(sequence):
        for j, nucleotide in enumerate(seq):
            if nucleotide == "N":
                # In case of wildcards, we assign equal probability to each nucleotide
                one_hot_sequence[i, j, 0] = 0.25
                one_hot_sequence[i, j, 1] = 0.25
                one_hot_sequence[i, j, 2] = 0.25
                one_hot_sequence[i, j, 3] = 0.25
            elif nucleotide == "R":
                one_hot_sequence[i, j, 0] = 0.5
                one_hot_sequence[i, j, 1] = 0.5
            elif nucleotide == "Y":
                one_hot_sequence[i, j, 2] = 0.5
                one_hot_sequence[i, j, 3] = 0.5
            elif nucleotide == "K":
                one_hot_sequence[i, j, 0] = 0.5
                one_hot_sequence[i, j, 2] = 0.5
            elif nucleotide == "M":
                one_hot_sequence[i, j, 1] = 0.5
                one_hot_sequence[i, j, 3] = 0.5
            elif nucleotide == "S":
                one_hot_sequence[i, j, 0] = 0.5
                one_hot_sequence[i, j, 3] = 0.5
            elif nucleotide == "W":
                one_hot_sequence[i, j, 1] = 0.5
                one_hot_sequence[i, j, 2] = 0.5
            elif nucleotide == "B":
                one_hot_sequence[i, j, 0] = 1 / 3
                one_hot_sequence[i, j, 2] = 1 / 3
                one_hot_sequence[i, j, 3] = 1 / 3
            elif nucleotide == "D":
                one_hot_sequence[i, j, 0] = 1 / 3
                one_hot_sequence[i, j, 1] = 1 / 3
                one_hot_sequence[i, j, 2] = 1 / 3
            elif nucleotide == "H":
                one_hot_sequence[i, j, 1] = 1 / 3
                one_hot_sequence[i, j, 2] = 1 / 3
                one_hot_sequence[i, j, 3] = 1 / 3
            elif nucleotide == "V":
                one_hot_sequence[i, j, 0] = 1 / 3
                one_hot_sequence[i, j, 1] = 1 / 3
                one_hot_sequence[i, j, 3] = 1 / 3
            elif nucleotide == "-":
                one_hot_sequence[i, j, 0] = 0
                one_hot_sequence[i, j, 1] = 0
                one_hot_sequence[i, j, 2] = 0
                one_hot_sequence[i, j, 3] = 0
            else:
                one_hot_sequence[i, j, nucleotide_to_index[nucleotide]] = 1
    return one_hot_sequence


def test_model(path: str, test_sample: str):
    device = get_device()
    model = Blaster(
        input_size, max_chunked_sequence_length, len(classes), model_name
    ).to(device)
    checkpoint = load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # model.load_state_dict(load(path))
    model.eval()
    model_readable_sample = sequence_to_model(test_sample).to(device)
    return index_to_class_result_mapper(model(model_readable_sample))


if __name__ == "__main__":
    model_path = "models/Blaster_Bba162b_1699360752.pth"
    test_sample = "ATGAGTAAAAAAATACTTTATGGCAAAGAAGCAAGAAAAGCTTTATTACAAGGAGTAGATGCAATTGCAAATACTGTTAAAGTGACTTTAGGACCTAAAGGACGTAATGTTATTTTAGAAAAAGCCTATGATTCACCTGCTATTGTAAATGATGGTGTTTCTATTGCTAAAGAAATTGAATTAAAAAATCCTTATCAAAATATGGGAGCAAAGTTAGTATATGAAGTAGCTTCCAAAACTAACGATAAAGCAGGAGATGGAACAACTACAGCAACTGTTTTGGCACAAAGTATGATTCATCGTGGGTTTGATGCAATTGATGCAGGAGCTAATCCTGTTTTAGTAAAAGAAGGAATTGAGTTAGCTGCATTAACAGTTGCCAAAAAACTTTTAGCTAAATCTAAAAAAGTAGACGCCCAAGAAGATATTCAAAATGTGGCTGCTGTTTCATCAGGTAGTCAAGAAATTGGTAAAATCATTGCCCAAGCGATGCAAAAAGTAGGAAAAGATGGAGTTATTAATGTTGATGAATCCAAAGGTTTTGAAACAGAATTAGAAGTTGTTGAAGGATTGCAGTACGATAAAGGATATGCTTCTCCTTATTTTGTCTCTGATAGAGAAAGTATGACAGTACAGTTAGAAAATGCGTTAGTTTTAGTAACTGATCATAAAATTAGTACTGTGCAAGAAATTGTACCTATTTTGGAAGAAGTAGTAAAAGCATCTAGACCTTTATTAATTGTAGCTGAAGCTGTGGAAAATGAAGTTTTAGGGGTTTTGGTAGCTAATAAATTAAGAGGAACTTTTAATGTAGTTGTAACTAATGCTCCTGGTTTTGGTGATAATCAAAAAGAAATGTTACAAGATATTGCAGTACTTACAAAAGCTAATTTTGTTTCTAAAGAACTTAATATGAAATTAGCAGATTTAAAAATGGATGATTTAGGAAATATCAATAAAGCTATTATTAAAAAAGATAATACTACTTTGATAAGTAATTCTAAAAGTCCTGAATTAGAAAAACGTATTCAAGTATTAAAAACTCAAATTAAAAATGCTACTTCTGATTATGAAACTAAAAATTTGCAAGAAAGATTAGCTAAATTATCAGGAGGAGTTGCCTTAATTAAAGTTGGGGCTGCAACTGATACTGAATTAAAAGATAAAAAATTACGTATTGAAGATGCTCTTAATGCTACTAAAGCTGCTATTACTGAAGGAATTGTAGTTGGTGGTGGAAAAGCTTTAGTTGAGGTTTATCAAGAATTAAAAGATACTTTGGTATCTGATAATAAAGAAGTACAACAAGGAATTGATGTAGTAGTACAAAGTCTTTTAGTACCTACTTATCAAATTGCTTATAATGCAGGATTTTCAGGTAAAGATGTGGTAAAACAACAACTTTTACAACCCTTAAATTTTGGGTTTAATGCTAAAGAAGGTAAGTATGTTTGTCTCTTAAAAGAAGGTATAATTGACCCTACTAAAGTAACTCGTCAAGCTGTTCTTAATGCTGCTTCTATCTCTGCTTTGATGATTACAACTGAGGCTGCTGTGGTAAGTTTGAAAGAAAATAAAGATAATAACTTTGATTTAGGAACACAAGAATAA"
    result = test_model(model_path, test_sample)
    print(result)
