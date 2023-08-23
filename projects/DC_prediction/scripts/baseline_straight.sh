gpu_num=$1

cd /opt/railroad/projects/DC_prediction

epoch=200
BS=4
interval=50

rail_type=straight
val_type=pre
window_length=2500
history_length=500
in_planes=8
f_maps=16
kernel=5
dilation=2

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=baseline-${val_type}-interval${interval}-${rail_type}-window-${window_length}-${history_length}-inplane${in_planes}-fmaps${f_maps}-k${kernel}-d${dilation} \
  loader.batch_size=${BS} \
  loader.dataset.val_type=${val_type} \
  loader.dataset.window_length=${window_length} \
  loader.dataset.history_length=${history_length} \
  loader.dataset.interval=${interval} \
  loader.dataset.rail_type=${rail_type} \
  model.in_planes=${in_planes} \
  model.f_maps=${f_maps} \
  model.kernel=${kernel} \
  model.dilation=${dilation} \
  scheduler.max_lr=1e-03 \
  trainer.max_epoch=${epoch} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False
