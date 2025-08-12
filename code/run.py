import os
import random
import torch
import numpy as np

from params import get_parsed_args
from utils import save_config, get_logger
from evaluation import evalrank
from train import main

def run():

    opt =get_parsed_args()

    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    if not opt.noise_file:
        opt.noise_file = os.path.join(
            opt.output_dir, opt.data_name + "_" + str(opt.noise_ratio) + ".npy"
        )

    if opt.data_name == "cc152k_precomp":
        opt.noise_ratio = 0
        opt.noise_file = ""

    print("\n*-------- Experiment Config --------*")
    print(opt)

    # CUDA env
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    # set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True

    # save config
    save_config(opt, os.path.join(opt.output_dir, "config.json"))

    print("\n*-------- 模型训练阶段 --------*")
    global logger
    logger = get_logger(os.path.join(opt.output_dir, "log.txt"))

    main(opt, logger, resume=False)


    logger.info("\n*-------- Testing Best Epoch --------*")
    if opt.data_name == "coco_precomp":
        logger.info("5 fold validation")
        evalrank(
            os.path.join(opt.output_dir, "model_best.pth.tar"),
            logger,
            split="testall",
            fold5=True,
        )
        logger.info("full validation")
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), logger, split="testall")
    else:
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), logger, split="test")

    logger.info("\n*-------- Testing Last Epoch --------*")
    last_epoch_model_path = "checkpoint_" + str(opt.num_epochs - 1) + ".pth.tar"
    if opt.data_name == "coco_precomp":
        logger.info("5 fold validation")
        evalrank(
            os.path.join(opt.output_dir, last_epoch_model_path),
            logger,
            split="testall",
            fold5=True,
        )
        logger.info("full validation")
        evalrank(os.path.join(opt.output_dir, last_epoch_model_path), logger, split="testall")
    else:
        evalrank(os.path.join(opt.output_dir, last_epoch_model_path), logger, split="test")


if __name__ == "__main__":
    run()









