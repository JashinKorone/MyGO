"""Experiment entry point.

``quick_start`` merges the configuration files, builds the FakeSV+ dataloaders
and sweeps the hyper-parameter grid declared by ``hyper_parameters``.
"""

import os
import platform
from itertools import product
from logging import getLogger

from torch.utils.data import DataLoader

from utils.configurator import Config
from utils.dataloader import _init_fn, collate_fn
from utils.dataset import FVDDataset
from utils.logger import init_logger
from utils.utils import dict2str, get_model, get_trainer, init_seed


def build_dataloader(dataset, config, shuffle):
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['device'].type == 'cuda',
        shuffle=shuffle,
        worker_init_fn=_init_fn,
        collate_fn=collate_fn,
    )


def quick_start(model, dataset, config_dict):
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info('Server: \t' + platform.node())
    logger.info('Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # ------------------------------------------------------------------- data
    dataset = FVDDataset(config)
    config['num_events'] = dataset.num_events
    logger.info('\n====Training====\n' + str(dataset.train_dataset))
    if config['dataset_mode'] == 'time':
        logger.info('\n====Validation====\n' + str(dataset.val_dataset))
    logger.info('\n====Testing====\n' + str(dataset.test_dataset))

    train_dataloader = test_dataloader = val_dataloader = None
    if config['dataset_mode'] == 'event':
        # Event-split has no validation set, model selection uses the test set
        # following the 5-fold protocol of the original FakeSV benchmark.
        val = False
    else:
        val = True

    # -------------------------------------------------------- hyper-parameters
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    hyper_ls = []
    if 'seed' not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for name in config['hyper_parameters']:
        value = config[name]
        if value is None:
            value = [None]
        elif not isinstance(value, (list, tuple)):
            value = [value]
        hyper_ls.append(list(value))
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    last_batch_size = None
    for hyper_tuple in combinators:
        for name, value in zip(config['hyper_parameters'], hyper_tuple):
            config[name] = value
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        # ``batch_size`` may be part of the sweep (it is co-tuned with the
        # prompt candidate number K in Sec. 4.9), so the loaders are rebuilt
        # whenever it changes.
        if config['batch_size'] != last_batch_size:
            last_batch_size = config['batch_size']
            train_dataloader = build_dataloader(dataset.train_dataset, config, shuffle=True)
            test_dataloader = build_dataloader(dataset.test_dataset, config, shuffle=False)
            val_dataloader = (
                build_dataloader(dataset.val_dataset, config, shuffle=False) if val else None
            )

        model_instance = get_model(config['model'])(config, dataset.debunk_dataset)
        model_instance = model_instance.to(config['device'])
        trainer = get_trainer()(config, model_instance)
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
            train_data=train_dataloader,
            valid_data=val_dataloader,
            test_data=test_dataloader,
            saved=config['save_model'],
            val=val,
        )
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
                                                        hyper_ret[best_test_idx][0],
                                                        dict2str(hyper_ret[best_test_idx][1]),
                                                        dict2str(hyper_ret[best_test_idx][2])))

    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(
            config['hyper_parameters'], p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(
        config['hyper_parameters'], hyper_ret[best_test_idx][0],
        dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))
    return hyper_ret[best_test_idx]
