gpu_num=0
cd /opt/railroad/projects/DC_prediction

exp_name=alpha-test

epoch=30
interval=50
window_length=500
history_length=0
val_type=pre

mode=uniform
max_sigma=0.1
sigma_normal_scale=2.0

BS=4
LR=1e-2
WD=1e-4
alpha_mse=1.0

in_planes=8
f_maps=64
num_levels=4
kernel=5
dilation=2

rail_type=curved
HYDRA_FULL_ERROR=1 python3 main.py \
    experiment_tool.experiment_name=railroad-chpark \
    experiment_tool.run_group=${exp_name} \
    experiment_tool.run_name=${exp_name}-${rail_type}-f${f_maps}-alpha${alpha_mse}-linear \
    optim.weight_decay=${WD} \
    loader.batch_size=${BS} \
    loader.dataset.val_type=${val_type} \
    loader.dataset.window_length=${window_length} \
    loader.dataset.history_length=${history_length} \
    loader.dataset.interval=${interval} \
    loader.dataset.augmentation.gaussian_smoothing.mode=${mode} \
    loader.dataset.augmentation.gaussian_smoothing.max_sigma=${max_sigma} \
    loader.dataset.augmentation.gaussian_smoothing.sigma_normal_scale=${sigma_normal_scale} \
    "loader.dataset.rail_type=${rail_type}" \
    model.in_planes=${in_planes} \
    model.f_maps=${f_maps} \
    model.num_levels=${num_levels} \
    model.kernel=${kernel} \
    model.dilation=${dilation} \
    scheduler.max_lr=${LR} \
    criterion.alpha_mse=${alpha_mse} \
    trainer.max_epoch=${epoch} \
    trainer.gpus=${gpu_num} \
    trainer.fast_dev_run=False
