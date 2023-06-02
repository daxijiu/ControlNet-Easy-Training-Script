import os
from share import *
from config import *
import warnings
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

import time
from datetime import datetime, timedelta

torch.set_float32_matmul_precision('medium')

# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = config.batch_size
not_logger = config.not_logger
logger_freq = config.logger_freq
learning_rate = config.learning_rate
max_steps = config.max_steps
max_epochs = config.max_epochs
save_ckpt_every_n_steps = config.save_ckpt_every_n_steps
save_top_k = config.save_top_k
save_weights_only = config.save_weights_only
save_last = config.save_last
sd_locked = config.sd_locked
only_mid_control = config.only_mid_control

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# ckpt_callback
checkpoint_callback = ModelCheckpoint(
                dirpath='./output/',
                every_n_train_steps=save_ckpt_every_n_steps,
                save_weights_only=save_weights_only,
                save_top_k=save_top_k,
                filename='my_controlnet_sd21_{epoch:03d}_{step:06d}_{val_loss:.4f}',
                save_last=save_last
            )

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, disabled=not_logger)
trainer = pl.Trainer(accelerator='gpu', devices='auto', precision=32, max_steps=max_steps, max_epochs=max_epochs, callbacks=[logger, checkpoint_callback])

# Ignore warnings
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")
warnings.filterwarnings("ignore", ".*The dataloader, train_dataloader, does not have many workers which may be a bottleneck*")
warnings.filterwarnings("ignore", ".*You defined a `validation_step` but have no `val_dataloader`. Skipping val loop*")
warnings.filterwarnings("ignore", ".*in your `training_step` but the value needs to be floating point. Converting it to torch.float32*")

# Train!
t_start =time.time()
print('训练开始')
trainer.fit(model, dataloader)
sec = timedelta(seconds=int(time.time()-t_start))
d = datetime(1,1,1) + sec
print('训练结束')
print("本次训练耗时%d天%d小时%d分钟%d秒" % (d.day-1, d.hour, d.minute, d.second))