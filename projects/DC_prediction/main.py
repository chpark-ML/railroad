import logging

import hydra
import omegaconf

from train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path='configs', config_name='config')
def main(config: omegaconf.DictConfig) -> None:
    logger.info("Training DC prediction Model.")
    return train(config)


if __name__ == '__main__':
    main()
