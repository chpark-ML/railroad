gpu_num=$1

cd /opt/railroad/projects/DC_prediction

val_type=pre

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=debug \
  experiment_tool.run_name=debug \
  loader.dataset.val_type=${val_type} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=True