# F74092031_GAI_Project4
本作業以 Anaconda Python 3.8 + CUDA 11.8 環境運行。

要運行 DIP 的初始先驗加速 DDPM，請執行
> python dip_init_ddpm.py

要運行 DDPM 啟發的監督引導 DIP 提前停止，請執行
> python dip_earlystopping.py

要運行原版 DIP，請執行
> python dip.py

要運行原版 DDPM，請執行
> python ddpm/training_model.py