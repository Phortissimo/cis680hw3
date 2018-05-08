import numpy as np                                                                                    
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader, load
from utils import prepare_dirs_and_logger, save_config

def main(config):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  load()
  train_data_loader, train_label_loader, train_loc_loader, train_mask_loader = get_loader(config.data_path, config.batch_size, 0, 'train', True)
  test_data_loader, test_label_loader, test_loc_loader, test_mask_loader = get_loader(config.data_path, config.batch_size_test, 5, 'train', True)

  trainer = Trainer(config, train_data_loader, train_label_loader, train_loc_loader, train_mask_loader, test_data_loader, test_label_loader, test_loc_loader, test_mask_loader)
  print("loaded trainer")
  if config.is_train:
    save_config(config)
    trainer.train()
    print("finished train")
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

if __name__ == "__main__":
  config, unparsed = get_config()
  main(config)
