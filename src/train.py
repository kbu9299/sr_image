import argparse
import json
from sr_model import SrModel


def main(config):
    model = SrModel(config)
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SrModel')
    parser.add_argument('--config', default='config/config.json', type=str, help='config file')
    args = parser.parse_args()

    with open(args.config, "r") as read_file:
        config = json.load(read_file)

    main(config)
