import argparse
import json
import os

from sr_model import SrModel
from utils import load_data


def main(config, checkpoint):
    sr_model = SrModel(config)
    sr_model.load_checkpoint(checkpoint)

    train_dataset, val_dataset, test_dataset, baseline_cpsnrs = \
        load_data(config, seq_len=config['training']['seq_len'], val_proportion=0.10)

    results = sr_model.evaluate(train_dataset, val_dataset, test_dataset, baseline_cpsnrs)

    print("\nBenchmark % ESA baseline")
    print(results.describe().T)

    print("\nTrain")
    print(results.loc[results['part'] == 'train'].describe().loc['mean'])

    print("\nValidation")
    print(results.loc[results['part'] == 'val'].describe().loc['mean'])

    print("\nTest")
    print(results.loc[results['part'] == 'test'].describe().loc['mean'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Super-Resolved Evaluation')
    parser.add_argument('--config', default='config/config.json', type=str, help='config file')
    parser.add_argument('--checkpoint_file', required=True, type=str, help='checkpoint file')
    args = parser.parse_args()

    with open(args.config, "r") as read_file:
        config = json.load(read_file)

    assert os.path.isfile(args.checkpoint_file)

    main(config, args.checkpoint_file)
