gpu_num=$1

cd /opt/railroad/projects/DC_prediction

epoch=500
BS=4
interval=50

rail_type=curved
val_type=post
window_length=4000
history_length=2000
in_planes=16
f_maps=32

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=baseline-${val_type}-interval${interval}-${rail_type}-window-${window_length}-${history_length}-inplane${in_planes}-fmaps${f_maps} \
  loader.batch_size=${BS} \
  loader.dataset.val_type=${val_type} \
  loader.dataset.window_length=${window_length} \
  loader.dataset.history_length=${history_length} \
  loader.dataset.interval=${interval} \
  loader.dataset.rail_type=${rail_type} \
  model.in_planes=${in_planes} \
  model.f_maps=${f_maps} \
  scheduler.max_lr=1e-03 \
  trainer.max_epoch=${epoch} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False