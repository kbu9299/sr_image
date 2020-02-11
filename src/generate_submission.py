import argparse
import json
import os

from sr_model import SrModel


def main(config, checkpoint_file, out):
    sr_model = SrModel(config)
    sr_model.generate_submission(checkpoint_file, out=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate files for submission')
    parser.add_argument('--config', default='config/config.json', type=str, help='config file')
    parser.add_argument('--checkpoint_file', required=True, type=str, help='checkpoint file')
    parser.add_argument('--submission_dir', default='submission', type=str, help='submission folder')

    args = parser.parse_args()

    with open(args.config, "r") as read_file:
        config = json.load(read_file)

    assert os.path.isfile(args.checkpoint_file)

    main(config, args.checkpoint_file, args.submission_dir)
