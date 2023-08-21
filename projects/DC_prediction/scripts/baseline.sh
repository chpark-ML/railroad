gpu_num=$1

cd /opt/railroad/projects/DC_prediction

epoch=1000
BS=16

HYDRA_FULL_ERROR=1 python3 main.py \
  experiment_tool.experiment_name=railroad-chpark \
  experiment_tool.run_group=debug \
  experiment_tool.run_name=debug \
  loader.batch_size=${BS} \
  trainer.max_epoch=${epoch} \
  trainer.gpus=${gpu_num} \
  trainer.fast_dev_run=False