import torch

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config
import pdb

def main(config):
    prepare_dirs_and_logger(config)
    #pdb.set_trace()
    torch.manual_seed(config.random_seed)
    if config.num_gpu > 0:
        torch.cuda.manual_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
    else:
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image

    data_path = './NUS-WIDE/'
    data_loader = get_loader(
            data_path, batch_size, config.input_scale_size,
            config.num_worker, ifshuffle=True, TEST=False, FEA=True)

    test_data_loader = get_loader(
            data_path, batch_size, config.input_scale_size,
            config.num_worker, ifshuffle=False, TEST=True, FEA=True)

    torch.cuda.set_device(5)
    trainer = Trainer(config, data_loader, test_data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
