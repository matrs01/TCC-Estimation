import yaml
import os

import wandb


def main():
    with open("configs/shad.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    if not config["wandb_on"]:
        os.environ["WANDB_MODE"] = "offline"

    wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device_index']

    from trainer.trainer import Trainer

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
