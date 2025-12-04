import os
import argparse
from utils.quick_start import quick_start
import warnings

os.environ['NUMEXPR_MAX_THREADS'] = '48'
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DIF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='FakeSV-50', help='name of datasets')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict)
