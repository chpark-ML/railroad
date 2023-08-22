gpu_num=$1

cd /opt/railroad/projects/DC_prediction

rail_type=curved

HYDRA_FULL_ERROR=1 python3 analysis/test.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=test \
  loader.dataset.rail_type=${rail_type} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False