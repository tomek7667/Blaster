
classes = ['Mycoplasmatota', 'Firmicutes', 'Deinococcus-Thermus', 'Fusobacteriota', 'Myxococcota', 'Thermotogae', 'Spirochaetes', 'Bacteroidota', 'Chlorobi', 'Tenericutes', 'Verrucomicrobia', 'Cyanobacteria', 'Chlamydiae', 'Fusobacteria', 'Actinomycetota', 'Campylobacterota', 'Planctomycetota', 'Proteobacteria', 'Bacillota', 'Fibrobacteres', 'Actinobacteria', 'Pseudomonadota', 'Aquificota', 'Chloroflexi']
sequence_length = 700
num_classes = 24
dropout = 0
lstm_layers = 3
is_bidirectional = True
first_layer_chunk = 8
should_use_softmax = True
b_size = 128
batch_size = 16
optimizer = "adam"
learning_rate = 0.0001

# Net layers:# Layer: embedding.weight | Size: torch.Size([350, 2800]) | Num el: 980000
# Layer: embedding.bias | Size: torch.Size([350]) | Num el: 350
# Layer: lstm1.weight_ih_l0 | Size: torch.Size([512, 350]) | Num el: 179200
# Layer: lstm1.weight_hh_l0 | Size: torch.Size([512, 128]) | Num el: 65536
# Layer: lstm1.bias_ih_l0 | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.bias_hh_l0 | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.weight_ih_l0_reverse | Size: torch.Size([512, 350]) | Num el: 179200
# Layer: lstm1.weight_hh_l0_reverse | Size: torch.Size([512, 128]) | Num el: 65536
# Layer: lstm1.bias_ih_l0_reverse | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.bias_hh_l0_reverse | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.weight_ih_l1 | Size: torch.Size([512, 256]) | Num el: 131072
# Layer: lstm1.weight_hh_l1 | Size: torch.Size([512, 128]) | Num el: 65536
# Layer: lstm1.bias_ih_l1 | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.bias_hh_l1 | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.weight_ih_l1_reverse | Size: torch.Size([512, 256]) | Num el: 131072
# Layer: lstm1.weight_hh_l1_reverse | Size: torch.Size([512, 128]) | Num el: 65536
# Layer: lstm1.bias_ih_l1_reverse | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.bias_hh_l1_reverse | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.weight_ih_l2 | Size: torch.Size([512, 256]) | Num el: 131072
# Layer: lstm1.weight_hh_l2 | Size: torch.Size([512, 128]) | Num el: 65536
# Layer: lstm1.bias_ih_l2 | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.bias_hh_l2 | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.weight_ih_l2_reverse | Size: torch.Size([512, 256]) | Num el: 131072
# Layer: lstm1.weight_hh_l2_reverse | Size: torch.Size([512, 128]) | Num el: 65536
# Layer: lstm1.bias_ih_l2_reverse | Size: torch.Size([512]) | Num el: 512
# Layer: lstm1.bias_hh_l2_reverse | Size: torch.Size([512]) | Num el: 512
# Layer: linear1.weight | Size: torch.Size([24, 256]) | Num el: 6144
# Layer: linear1.bias | Size: torch.Size([24]) | Num el: 24

# Total parameters: 2268566