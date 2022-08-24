from bam_poses.eval.evaluation import Evaluation
import bam_poses.eval.utils as eval_utils
from bam_poses.transforms.transforms import normalize, undo_normalization_to_seq
from models.SeSGCN_teacher import Model

import torch
import numpy as np

TMP_DIR = "/home/ssg1002/BAM_cache"  # this path needs ~125GB of free storage

EVAL_DATASET = "D"  # choose the evaluation dataset
ev = Evaluation(
    dataset=EVAL_DATASET,
    tmp_dir=TMP_DIR,
    n_in=10,
    n_out=250,
    data_location="/home/ssg1002/Datasets/BAM/0_1_0",
    default_to_box
    =True
)
# Important! Creating the evaluation for the first time will take a considerable amount
# of time (potentially ~2h) - make sure to provide a TMP_DIR where you can store the
# data for future use!

input_n = 10 # number of frames to train on(default=10)
output_n = 25 # number of frames to predict on
input_dim = 3 # dimensions of the input coordinates(default=3)
joints_to_consider = 17  #joints

# FLAGS FOR THE MODEL
tcnn_layers = 4 # number of layers for the Temporal Convolution of the Decoder (default=4)
tcnn_kernel_size = [3, 3] # kernel for the T-CNN layers (default=[3,3])
st_gcnn_dropout = 0.1 # (default=0.1)
tcnn_dropout = 0.0  # (default=0.0)

model = Model(input_dim, input_n, output_n, st_gcnn_dropout, joints_to_consider, tcnn_layers, tcnn_kernel_size, tcnn_dropout)

model_pth = "/home/ssg1002/cvg_hiwi/CHICO-PoseForecasting/checkpoints/BAM_3d_25_frames_ckpt_STS"
state_dict = torch.load(model_pth)
model.load_state_dict(state_dict, strict=True)
model.eval()
model.cuda()

def fn(persons_in, masks_in, scene, frame, n_in, n_out, pids):
    """
    Callback for generating the results. Your model predicts the
    data in here.
    :param persons_in: {n_persons x n_in x 17 x 3}
    :param masks_in: {n_persons x n_in}
    :param scene: {bam_poses.data.scene.Scene}
    :param frame: {int}
    :param n_in: {int}
    :param n_out: {int}
    :param pids: {List[int]}
    """
    # note that we don't batch the data. Before passing to the
    # model you will have to "batch" your data:
    
    persons_out = []
    for person in persons_in:
        normalized_seq, normalization_params = normalize(person, -1, return_transform=True)
        input_seq = torch.Tensor(normalized_seq).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        person_out = []
        for n_iter in range(n_out // output_n):
            output = model(input_seq)
            output = output.permute(0, 1, 3, 2)
            person_out_hat = undo_normalization_to_seq(
                    output[0].cpu().detach().numpy(),
                    normalization_params[0],
                    normalization_params[1]
                )
            person_out.append(person_out_hat)
            person = person_out_hat[-input_n:]
            normalized_seq, normalization_params = normalize(person, -1, return_transform=True)
            input_seq = torch.Tensor(normalized_seq).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        person_out = np.concatenate(person_out, 0)
        persons_out.append(person_out)
    persons_out_hat = np.array(persons_out)
    return persons_out_hat.astype(np.float64)  # note that we have to "unbatch"

# run the evaluation
result = ev.ndms(fn)

# save results to file
eval_utils.save_results(TMP_DIR + "/results_CHICO.pkl", result)