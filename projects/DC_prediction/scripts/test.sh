gpu_num=$1

cd /opt/railroad/projects/DC_prediction

BS=5

HYDRA_FULL_ERROR=1 python3 analysis/test.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=test \
  loader.dataset.rail_type=${rail_type} \
  loader.batch_size=${BS} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False

HYDRA_FULL_ERROR=1 python3 analysis/val.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=val \
  loader.dataset.rail_type=${rail_type} \
  loader.batch_size=${BS} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False