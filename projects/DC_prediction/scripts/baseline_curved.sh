gpu_num=$1

cd /opt/railroad/projects/DC_prediction

epoch=30
BS=8
interval=5
rail_type=curved

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=baseline-interval${interval}-${rail_type} \
  loader.batch_size=${BS} \
  loader.dataset.interval=${interval} \
  loader.dataset.rail_type=${rail_type} \
  trainer.max_epoch=${epoch} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False