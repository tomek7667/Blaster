from torch import load
from model import Blaster, get_device
from models.ba26ed import *


def test_model(path: str, test_sample: str):
    device = get_device()
    model = Blaster(
        input_size, max_chunked_sequence_length, len(classes), model_name
    ).to(device)
    model.load_state_dict(load(path))
    model.eval()
    print(model(test_sample))


if __name__ == "__main__":
    model_path = "models/Blaster_ba26ed_1699346446.pth"
    test_sample = "CTGCAGCCGCCGACTGAAATCTATCGGGAAGAAAAGCTCGCTTACGACACCTTTAACCCGCAGGATCCAGTCGCTTACCTCGCATCTCAAAAGCAGAAATACGGGAGATAAACACAACTTATGGTGAGAACTCCTGTACCGCTTTACCTACGTTGGGCGGTCTCCATCCTCAGCGTGCTTGCGTTCCTAGCCATTTGGCAAATTGCGGCAGCTTCAGGATTTTTAGGCAAAACTTTTCCTGGCTCCCTGCGCACTTTGCAGGATTTGTTTGGATGGCTTTCAGATCCCTTCTTTGATAACGGCCCCAATGACTTAGGGATTGGCTGGAACTTACTGATTAGTTTGCGTCGCGTTGCGATCGGCTACCTGCTGGCAACAGTTGTTGCAATTCCTTTGGGGATTGCAATCGGTATGTCGGCGCTAGCTTCCAGTATTTTTTCGCCCTTTGTGCAACTCCTGAAGCCAGTTTCACCTTTGGCCTGGTTGCCGATTGGTCTCTTCTTATTCCGAGATTCGGAATTGACGGGTGTTTTTGTCATCCTGATTTCGAGTCTGTGGCCAACGTTGATCAACACAGCGTTTGGGGTGGCGAATGTCAATCCTGACTTTTTGAAGGTTTCGCAATCTTTGGGAGCTAGTCGTTGGCGCACGATTCTGAAGGTGATTCTGCCCGCAGCATTGCCCAGCATCATCGCGGGAATGCGGATCAGCATGGGCATTGCTTGGCTGGTCATTGTGGCAGCAGAGATGCTGTTGGGAACAGGAATTGGCTATTTCATTTGGAATGAGTGGAATAACCTATCACTTCCTAATATTTTCTCGGCCATCATCATCATTGGGATTGTTGGCATTCTTCTCGACCAAGGCTTCCGTTTTCTTGAGAACCAGTTTTCTTACGCAGGCAACCGATAACCCATGATTTCTGAAGCTGTGCCAGCCAAGGAGGAGACAGGGCAGGCTCAATTGCTGATTGAGCAAGTTGGCAAAGTTTTTACTGTCAATTCACCTTCTCTCCTCGATCGCCTTCGACAGCGATCGCCCAAACGCTACGTTGCATTAGAAGATGTCAACCTCACGATCGCGTCGAACACATTTGTCTCGATTATTGGCCCTTCGGGTTGTGGTAAATCAACCCTTCTCAACTTGATTGCTGGCCTTGATTTACCAACGTCTGGCCAGATTCTGCTGGATGGTCAACGCATTCGATCGCCGGGGCCCGATCGTGGCATCGTCTTCCAGAACTATGCCCTGATGCCCTGGATGACCGCGCTTGAGAATGTCATCTTTGCAGTTGAAACGGCGCGCCCAAACCTGAGCAAATCCCAAGCTCGCGAAGTGGCACGAGAGCATCTAGAGCTGGTGGGTTTAACCAAAGCTGCCGATCGCTATCCGGGCCAAATTTCAGGGGGGATGAAACAGCGCGTAGCGATCGCCCGTGCCCTCTCCATCCGTCCTAAGCTCCTGCTGATGGATGAACCCTTTGGTGCCTTGGATGCCCTCACCCGTGGCTACCTCCAAGAAGAAGTGCTGCGGATTTGGGAAGCCAACAAACTGAGTGTGGTGCTCATCACTCACAGTATTGATGAAGCACTGCTGCTTTCCGATCGCATTGTGGTGATGTCTCGTGGGCCACGAGCCACTATTCGAGAAGTGATTGATTTACCAGCCGTTCGCCCTCGGCAACGGTCTGTGATCGAAGAAGATGAGCGCTTCGTCAAAATCAAATTGCGCCTTGAAGAACATTTGTTCAACGAGACGCGTGCAGTTGAAGAAGCCAGTGTTTAGGAGAATTCCAATGACCTCAGCGATTACTGAACAACTTCTGAAAGCGAAAAAAGCAAAGGGAATTACCTTTACTGAGCTTGAGCAATTACTTGGACGGGATGAAGTCTGGATTGCGAGTGTGTTCTACCGTCAATCTACGGCTTCGCCTGAAGAGGCAGAAAAGCTACTGACTGCTCTGGGCTTAGATCTGGCCTTGGCTGATGAGTTGACGACTCCGCCGGTCAAAGGTTGTTTGGAACCGGTGATTCCAACTGATCCGTTGATCTATCGCTTCTACGAAATCATGCAGGTCTATGGCTTGCCCCTCAAGGATGTTATCCAAGAAAAATTTGGCGATGGCATCATGAGTGCGATTGATTTCACCTTAGATGTCGATAAGGTTGAAGATCCCAAAGGCGATCGCGTTAAGGTCACGATGTGTGGCAAGTTCTTGGCGTACAAGAAGTGGTAAATACTGCTAGCTAATCAAGCTTCAATTCTTGATCACTGGAGGAGAGAGGTTTCCGCTTCTCTCCTTTTTTGATTGGAATTCTCTCATTAACTACGATACCGCTCTGCACTGAATGACCTCGAGCTGAGTGGAAGGTAGCTCGCCGCCGATGATAATGGCGCCTCTGGAAGAGTTTGGCTAAGCTGTGGACGGCGATCGCGGTTGTCTGTCTGTGCTATGCCCTTGATTTCGGTGACCCGACTCAAGCTTAGAAATGTTCTTTATTTGCCCCGCTTGCTTCCCTTCTCGTTGCGATCGACGTGGCAGGCTAAACGAGCGCCTGGCAATCTGGGCGTTAAGCTGTTGCAGGATCGTAACTTGGCTTTTTGGACCTGCACCGCTTGGACGGATGAAGGAGCCATGCGTCGGTTCATGAGAGCGGATGCCCACGGGCAGGCCATGACGAAATTGATGGATTGGTGCAGCGAAGCCTCAGTCGTCCATTGGCAGCAGGATCAGCCAGACTTGCCCGACTGGCAGGAAGCTCACCGCCGCATGATCGCGGAGGGGCGCCCCTCCAAAGTGAACCATCCTTCGGCTGCCCACCAAGCATTTCAGGTCGATCCGCCGCGCCGCGCCTAGCTCAGTGACTGCGGTCGCGCTGTCTTGCATCATTGCTTCGCTCTACCAGCCCGGATCGCTGGCACAGTCCACGGTGATCTCACCCGAGGCGGCATCGGGAATCGCAGTGATACAGCCGCAGACTGGCTCGCCATC"
    test_model(model_path, test_sample)