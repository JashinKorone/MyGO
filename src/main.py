"""MyGO -- Modality-incomplete Fake News Video Detection (TOMM 2025).

Examples::

    # MyGO on FakeSV+ with a global missing rate eta = 0.3
    python main.py --model MyGO --dataset FakeSV-30

    # ablation: -w/o PMA (Table 4)
    python main.py --model MyGO --dataset FakeSV-30 --use_pma False

    # plug PMA into the SVFEND baseline (Table 6)
    python main.py --model SVFEND --dataset FakeSV-30 --use_pma True
"""

import argparse
import os
import warnings

from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'
warnings.filterwarnings('ignore')


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected, got ' + str(value))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', '-m', type=str, default='MyGO',
                        help='MyGO | SVFEND | SimpleFusion')
    parser.add_argument('--dataset', '-d', type=str, default='FakeSV-30',
                        help='FakeSV (eta=0) or FakeSV-{10,30,50,70}')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=None,
                        help='paper searches within [1e-4, 5e-5, 1e-3]')
    parser.add_argument('--batch_size', type=int, default=None, help='b, default 256')
    parser.add_argument('--epochs', type=int, default=None)

    # loss weights, Eq. 14
    parser.add_argument('--ctrs_loss_wgt', type=float, default=None, help='alpha, default 0.3')
    parser.add_argument('--orth_loss_wgt', type=float, default=None, help='beta, default 0.2')
    parser.add_argument('--cl_temp', type=float, default=None, help='tau, default 0.2')

    # PMA, Sec. 3.4
    parser.add_argument('--prompt_top_k', type=int, default=None, help='K, default 10')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='default 5')
    parser.add_argument('--pma_weight', type=float, default=None, help='u of Eq. 13')

    # ablation switches, Table 4
    parser.add_argument('--use_pma', type=str2bool, default=None, help='-w/o PMA')
    parser.add_argument('--use_pma_loss', type=str2bool, default=None, help='-w/o PMA loss')
    parser.add_argument('--use_prompt', type=str2bool, default=None, help='drop missing prompts')
    parser.add_argument('--use_caption_weight', type=str2bool, default=None, help='-w/o CKA')
    parser.add_argument('--disentangle', type=str2bool, default=None, help='-w/o MDN')
    return parser


CONFIG_KEYS = (
    'gpu_id', 'learning_rate', 'epochs', 'ctrs_loss_wgt', 'orth_loss_wgt', 'cl_temp',
    'prompt_top_k', 'warmup_epochs', 'pma_weight', 'use_pma', 'use_pma_loss', 'use_prompt',
    'use_caption_weight', 'disentangle',
)
# ``seed`` and ``batch_size`` are swept as grids, hence wrapped into lists.
GRID_KEYS = ('seed', 'batch_size')


if __name__ == '__main__':
    args, _ = build_parser().parse_known_args()

    config_dict = {}
    for key in CONFIG_KEYS:
        value = getattr(args, key)
        if value is not None:
            config_dict[key] = value
    for key in GRID_KEYS:
        value = getattr(args, key)
        if value is not None:
            config_dict[key] = [value]

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict)
