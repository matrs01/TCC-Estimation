import yaml
import os

import wandb


def main():
    wandb.login()

    with open("configs/shad.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['device_index']

    from trainer.tester import Tester

    tester = Tester(config)
    tester.test()


if __name__ == '__main__':
    main()
