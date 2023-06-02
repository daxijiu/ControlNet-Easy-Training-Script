@ECHO off
CHCP 65001 >nul
SET PYTHONIOENCODING=utf8
SET SITE_FIX_TORCH_SAVE_ENCODING=2
SET HF_HOME=%~dp0.cache\huggingface
.\env\python.exe mergetxt.py >nul
.\env\python.exe tool_add_control_sd21.py ./models/basemodel_sd21.ckpt ./models/control_sd21_ini.ckpt
.\env\python.exe tutorial_dataset.py
.\env\python.exe tutorial_dataset_test.py
.\env\python.exe tutorial_train_sd21.py
pause