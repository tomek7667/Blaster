
classes = ['Fibrobacteres', 'Actinobacteria', 'Chlorobi', 'Cyanobacteria', 'Chlamydiae', 'Spirochaetes', 'Tenericutes', 'Bacillota', 'Chloroflexi', 'Fusobacteriota', 'Pseudomonadota', 'Planctomycetota', 'Thermotogae', 'Myxococcota', 'Proteobacteria', 'Fusobacteria', 'Campylobacterota', 'Verrucomicrobia', 'Firmicutes', 'Deinococcus-Thermus', 'Aquificota', 'Actinomycetota', 'Mycoplasmatota', 'Bacteroidota']
sequence_length = 700
num_classes = 24
dropout = 0.4
lstm_layers = 2
is_bidirectional = False
first_layer_chunk = 8
should_use_softmax = False
b_size = 512
batch_size = 16
optimizer = "adam"
learning_rate = 0.001

# Net layers:# Layer: embedding.weight | Size: torch.Size([350, 2800]) | Num el: 980000
# Layer: embedding.bias | Size: torch.Size([350]) | Num el: 350
# Layer: lstm1.weight_ih_l0 | Size: torch.Size([2048, 350]) | Num el: 716800
# Layer: lstm1.weight_hh_l0 | Size: torch.Size([2048, 512]) | Num el: 1048576
# Layer: lstm1.bias_ih_l0 | Size: torch.Size([2048]) | Num el: 2048
# Layer: lstm1.bias_hh_l0 | Size: torch.Size([2048]) | Num el: 2048
# Layer: lstm1.weight_ih_l1 | Size: torch.Size([2048, 512]) | Num el: 1048576
# Layer: lstm1.weight_hh_l1 | Size: torch.Size([2048, 512]) | Num el: 1048576
# Layer: lstm1.bias_ih_l1 | Size: torch.Size([2048]) | Num el: 2048
# Layer: lstm1.bias_hh_l1 | Size: torch.Size([2048]) | Num el: 2048
# Layer: linear1.weight | Size: torch.Size([24, 512]) | Num el: 12288
# Layer: linear1.bias | Size: torch.Size([24]) | Num el: 24

# Total parameters: 4863382

# BEST MODEL
