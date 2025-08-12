import os
import random
import argparse
import time


def get_parsed_args():

    # current_time
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--data_path", default="../../../dataSpace/p1/data", help="Path to datasets."
    )
    parser.add_argument(
        "--data_name", default="f30k_precomp", help="{coco,f30k,cc152k}_precomp"
    )
    parser.add_argument(
        "--vocab_path",
        default="../../../dataSpace/p1/vocab",
        help="Path to saved vocabulary json files.",
    )

    # ----------------------- training setting ----------------------#
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Size of a training mini-batch."
    )
    parser.add_argument(
        "--num_epochs", default=40, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr_update",
        default=10,
        type=int,
        help="Number of epochs to update the learning rate.",
    )
    parser.add_argument(
        "--learning_rate", default=0.0002, type=float, help="Initial learning rate."
    )
    # 默认是1 我改成8试试
    parser.add_argument(
        "--workers", default=8, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--log_step",
        default=200,
        type=int,
        help="Number of steps to print and record the log.",
    )
    parser.add_argument(
        "--grad_clip", default=2.0, type=float, help="Gradient clipping threshold."
    )
    parser.add_argument(
        "--temperature", default=0.07, type=float, help="Temperature factor for scaling similarity."
    )

    # 我打算：跑f30k数据集时，设为10和15
    # 跑coco时，因为一共只有20轮，所以设为5和10

    # 在我的实验中，就变成了clean_start_epoch  原来的值为10
    parser.add_argument(
        "--clean_start_epoch", default=10, type=int, help="Training epochs starting with hard samples."
    )
    # noisy_start_epoch  原来的值为15
    parser.add_argument(
        "--noisy_start_epoch", default=15, type=int, help="Training epochs starting with noisy samples."
    )

    # ------------------------- model setting -----------------------#
    parser.add_argument(
        "--img_dim",
        default=2048,
        type=int,
        help="Dimensionality of the image embedding.",
    )
    parser.add_argument(
        "--word_dim",
        default=300,
        type=int,
        help="Dimensionality of the word embeddloss_for_spliting.",
    )
    parser.add_argument(
        "--embed_size",
        default=1024,
        type=int,
        help="Dimensionality of the joint embedding.",
    )
    parser.add_argument(
        "--sim_dim", default=256, type=int, help="Dimensionality of the sim embedding."
    )
    parser.add_argument(
        "--num_layers", default=1, type=int, help="Number of GRU layers."
    )
    parser.add_argument("--bi_gru", action="store_false", help="Use bidirectional GRU.")
    parser.add_argument(
        "--no_imgnorm",
        action="store_true",
        help="Do not normalize the image embeddings.",
    )
    parser.add_argument(
        "--no_txtnorm",
        action="store_true",
        help="Do not normalize the text embeddings.",
    )
    parser.add_argument("--module_name", default="SGR", type=str, help="SGR, SAF")
    parser.add_argument("--sgr_step", default=3, type=int, help="Step of the SGR.")

    # noise settings
    parser.add_argument("--noise_file", default="", help="Noise index file.")
    parser.add_argument("--noise_ratio", default=0.2, type=float, help="Noisy ratio.")

    # ReCo Settings
    parser.add_argument(
        "--no_co_training", action="store_true", help="No co-training for noisy label."
    )
    parser.add_argument("--warmup_epoch", default=5, type=int, help="Warm up epochs.")

    #是否使用预warmup好的模型
    parser.add_argument("--warmup_model_path", default="", help="Warm up models.")
    parser.add_argument(
        "--p_threshold", default=0.5, type=float, help="Clean probability threshold."
    )
    parser.add_argument(
        "--loss_for_split", default="", help="Loss and probability for dividing the dataset."
    )
    parser.add_argument(
        "--soft_label", default=True, type=bool, help="Use soft label to train with infoNCE."
    )
    parser.add_argument(
        "--smooth_label", default=False, type=bool,
        help="Assign soft_label uniformly. Useless when soft_label == False."
    )
    parser.add_argument(
        "--draw_gmm", default=True, type=bool, help="Plot GMM of per sample loss."
    )

    # Runing Settings
    parser.add_argument("--gpu", default="3", help="Which gpu to use.")
    parser.add_argument(
        "--seed", default=random.randint(0, 100), type=int, help="Random seed."
    )

    #确定本次实验的输出目录路径，根据代码启动时间生成，输出文件位于./output/current_time下
    parser.add_argument(
        "--output_dir", default=os.path.join("output", current_time), help="Output dir."
    )
    # 新增一个名字参数，用来给wandb记录实验名字
    parser.add_argument(
        "--experiment_name", default=current_time, help="the experiment name."
    )

    opt=parser.parse_args()
    return opt