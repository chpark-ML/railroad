gpu_num=$1

cd /opt/railroad/projects/DC_prediction

epoch=30
BS=8
interval=5

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=baseline \
  experiment_tool.run_name=baseline-interva${interval} \
  loader.batch_size=${BS} \
  loader.dataset.interval=${interval} \
  trainer.max_epoch=${epoch} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False