from models.SeSGCN_teacher import Model
from tqdm import tqdm

from utils.haggling_eval import HagglingEvalDataset
from utils.haggling_utils import normalize, undo_normalization_to_seq

import torch
import numpy as np

input_n = 10 # number of frames to train on(default=10)
output_n = 25 # number of frames to predict on
input_dim = 3 # dimensions of the input coordinates(default=3)
joints_to_consider = 19  #joints

# FLAGS FOR THE MODEL
tcnn_layers = 4 # number of layers for the Temporal Convolution of the Decoder (default=4)
tcnn_kernel_size = [3, 3] # kernel for the T-CNN layers (default=[3,3])
st_gcnn_dropout = 0.1 # (default=0.1)
tcnn_dropout = 0.0  # (default=0.0)

model = Model(input_dim, input_n, output_n, st_gcnn_dropout, joints_to_consider, tcnn_layers, tcnn_kernel_size, tcnn_dropout)

model_pth = "checkpoints-haggling/haggling_3d_25_frames_ckpt_STS_best"
state_dict = torch.load(model_pth)
model.load_state_dict(state_dict, strict=True)
model.eval()
model.cuda()

sequences_in = HagglingEvalDataset(input_n, output_n)

sequences_out = []
n_seqs = len(sequences_in) // 3

for i in tqdm(range(n_seqs)):
    sequence_out_3_persons = []
    for p in range(3):
        sequence = sequences_in[(3 * i) + p]   # 178 x 19 x 3
        normalized_seq, normalization_params = normalize(sequence, -1, return_transform=True)
        for n_iter in range(1870 // output_n + 1):
            seq_in_torch = torch.from_numpy(normalized_seq[-input_n:]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            seq_out_torch = model(seq_in_torch).permute(0, 1, 3, 2)[0].cpu().detach()[-output_n:]
            seq_out = undo_normalization_to_seq(seq_out_torch.numpy(), normalization_params[0], normalization_params[1])
            sequence = np.concatenate((sequence, seq_out), 0)
            normalized_seq, normalization_params = normalize(sequence, -1, return_transform=True)
        sequence_out_3_persons.append(sequence)
    sequences_out.append(np.stack(sequence_out_3_persons, 1))

sequences_out = np.array(sequences_out)
print(sequences_out.shape)
np.save("haggling_eval_CHICO.npy", sequences_out[:, :2048])