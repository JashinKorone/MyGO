import platform
import os
from logging import getLogger
from itertools import product
from utils.configurator import Config
from utils.logger import init_logger
from utils.dataset import FVDDataset
from torch.utils.data import DataLoader
from utils.dataloader import collate_fn, _init_fn
from utils.utils import init_seed, get_model, get_trainer, dict2str


# from utils.logger import init_logger
# from utils.utils import init_seed, get_model, get_trainer, dict2str


def quick_start(model, dataset, config_dict):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('Server: \t' + platform.node())
    logger.info('Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = FVDDataset(config)
    config['num_events'] = dataset.num_events
    logger.info('\n====Training====\n' + str(dataset.train_dataset))
    if config['dataset_mode'] == 'time':
        logger.info('\n====Validation====\n' + str(dataset.val_dataset))
    logger.info('\n====Testing====\n' + str(dataset.test_dataset))

    # wrap into dataloader
    train_dataloader = DataLoader(dataset.train_dataset, batch_size=config['batch_size'],
                                  num_workers=config['num_workers'],
                                  pin_memory=True,
                                  shuffle=True,
                                  worker_init_fn=_init_fn,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=config['batch_size'],
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 shuffle=False,
                                 worker_init_fn=_init_fn,
                                 collate_fn=collate_fn)
    if config['dataset_mode'] == 'event':
        val_dataloader = None
        val = False
    else:
        val_dataloader = DataLoader(dataset.val_dataset, batch_size=config['batch_size'],
                                    num_workers=config['num_workers'],
                                    pin_memory=True,
                                    shuffle=False,
                                    worker_init_fn=_init_fn,
                                    collate_fn=collate_fn)
        val = True

    for batch_idx, batch_data in enumerate(train_dataloader):
        for k, v in batch_data.items():
            batch_data[k] = v.cuda()

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        ###Model###
        # model loading and initialization
        model = get_model(config['model'])(config, dataset.debunk_dataset).to(config['device'])
        # logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data=train_dataloader,
                                                                                valid_data=val_dataloader,
                                                                                test_data=test_dataloader,
                                                                                saved=config['save_model'],
                                                                                val=val)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
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

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                        hyper_ret[best_test_idx][0],
                                                                        dict2str(hyper_ret[best_test_idx][1]),
                                                                        dict2str(hyper_ret[best_test_idx][2])))
