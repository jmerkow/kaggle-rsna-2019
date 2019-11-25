import torch

from kaggle_lib.pytorch.ensembler import Ensembler
from kaggle_lib.pytorch.submission import median_filter_submissions

submission_headers_csv = '/data/rsna-intracranial-hemorrhage-detection/stage1-test-headers.csv'
model_root = '/mnt/nas/experiments-kaggle/rsna2019/'
dataset = 'rsna2019-stage1'
sub_len = 471270
models = [
    "amvepa/exp-nov1-resnext/r45_sca0.3_v0_wf1000_bs32_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0_w--m-subdural-bone-sm_bs32_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step006.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0_wf1000_bs64_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0.5_wf1000_bs32_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step006.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0_wf1000_bs32_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0_w--m-subdural-bone-sm_bs32_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-oct24-arvind/r45_wf1000_bs32_f0_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step006.pth.tar",
    "amvepa/exp-oct24-arvind/r45_w--m-subdural-bone-sm_bs32_f0_einceptionv4_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step006.pth.tar",
    "amvepa/exp-oct24-arvind/r45_wf1000_bs32_f0_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0_w--m-subdural-bone-sm_bs64_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/jtm_exp2-mini-coswr_march_highres_bs32_splitpatient/ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.3_v0_wf1000_bs32_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step006.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0_wf1000_bs32_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step006.pth.tar",
    "amvepa/exp-oct24-arvind/r45_w--m-subdural-bone-sm_bs32_f0_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.3_v0_w--m-subdural-bone-sm_bs64_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/exp-oct24-arvind/r45_w--m-subdural-bone-sm_bs32_f0_einceptionv4_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "amvepa/jtm_exp2-mini-coswr_march_highres_bs32_splitpatient/ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step002.pth.tar",
    "amvepa/exp-nov1-resnext/r45_sca0.4_v0_wf1000_bs64_f0_saNone_ese_resnext50_32x4d_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step006.pth.tar",
    "amvepa/exp-oct24-arvind/r45_w--m-subdural-bone-sm_bs64_f0_edensenet169_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar",
    "jameson/exp-nov1-inception/r45_sca0.3_v0_w--m-subdural-bone-sm_bs32_f0_saNone_einceptionv4_peTrue_T02_Tm1_em1e-08_tcos-restarts_soiTrue/checkpoints/model_step004.pth.tar"
]

print(models)

# If you leave out output_dir it will write to the model_root, which is fine.
e = Ensembler(model_root=model_root, output_dir=None, dataset=dataset)
print(e)

with torch.cuda.device(1):  # whatever device you want
    csv_file = e.ensemble(models, random_count=4)

print(csv_file)
submission = median_filter_submissions(csv_file, submission_headers_csv, r=5, avg=0.7, sub_len=sub_len)
