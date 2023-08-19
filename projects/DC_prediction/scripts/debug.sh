gpu_num=$1

cd /opt/railroad/projects/DC_prediction

epoch=10
BS=4
_gamma=2
_alpha=True
_smooth=0.0

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=debug \
  experiment_tool.run_name=debug \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=True