import os
import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


from vocab import deserialize_vocab
from data import get_dataset,get_loader
from model import SGRAF
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
    plot_gmm,
    save_loss_for_split,
)

def main(opt,logger,resume=False,checkpoint=None):
    logger.info("------------------实验基本信息----------------")
    logger.info(f"实验名称：{opt.experiment_name}")
    logger.info(f"使用的数据集：{opt.data_name}")
    logger.info(f"噪声率：{opt.noise_ratio}")
    logger.info(f"是否使用预warmup好的模型：{opt.warmup_model_path}")

    logger.info("-------------------------------------------")

    # load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        captions_train,
        images_train,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )

    val_loader = get_loader(
        captions_dev, images_dev, "dev", opt.batch_size, opt.workers
    )

    # create models
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    best_rsum = 0

    if not resume:
        best_rsum_warmup, best1, best2, best3, best4, best5, best6 = validate(opt, val_loader, logger,
                                                                              [model_A, model_B])
        best_warmup_epoch = 0

        # Warmup
        logger.info("* Warmup")
        # print("\n* Warmup")
        if opt.warmup_model_path:
            if os.path.isfile(opt.warmup_model_path):
                checkpoint = torch.load(opt.warmup_model_path)
                model_A.load_state_dict(checkpoint["model_A"])
                model_B.load_state_dict(checkpoint["model_B"])
                print(
                    "=> load warmup checkpoint '{}' (epoch {})".format(
                        opt.warmup_model_path, checkpoint["epoch"]
                    )
                )
            else:
                raise Exception(
                    "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
                )
        else:
            epoch = 0
            for epoch in range(0, opt.warmup_epoch):
                print("[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch))
                warmup(opt, noisy_trainloader, model_A, epoch)
                print("[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch))
                warmup(opt, noisy_trainloader, model_B, epoch)

                print("\nValidattion ...")
                rsum, b1, b2, b3, b4, b5, b6 = validate(opt, val_loader, logger, [model_A, model_B])

                is_best = rsum >= best_rsum_warmup and b1 >= best1 and b2 >= best2 and b3 >= best3 and b4 >= best4 and b5 >= best5 and b6 >= best6


                if is_best:
                    best_rsum_warmup = max(rsum, best_rsum_warmup)
                    best1 = max(b1, best1)
                    best2 = max(b2, best2)
                    best3 = max(b3, best3)
                    best4 = max(b4, best4)
                    best5 = max(b5, best5)
                    best6 = max(b6, best6)

                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "model_A": model_A.state_dict(),
                            "model_B": model_B.state_dict(),
                            "best_rsum": best_rsum_warmup,
                            "opt": opt,
                        },
                        is_best=False,
                        filename="warmup_model_{}.pth.tar".format(epoch),
                        prefix=opt.output_dir + "/",
                    )
                    best_warmup_epoch = epoch
                print("best warm up model is {}".format(
                    opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch)))

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "opt": opt,
                },
                is_best=False,  # this model is a warm up model
                filename="warmup_model_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

            if os.path.isfile(opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch)):
                checkpoint = torch.load(opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch))
                model_A.load_state_dict(checkpoint["model_A"])
                model_B.load_state_dict(checkpoint["model_B"])
                print(
                    "=> load warmup checkpoint '{}' (epoch {})".format(
                        opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch), checkpoint["epoch"]
                    )
                )
            else:
                print(
                    "=> no checkpoint found at '{}, no warming up is confirmed'".format(
                        opt.output_dir + "/" + "warmup_model_{}.pth.tar".format(best_warmup_epoch))
                )
                model_A = SGRAF(opt)
                model_B = SGRAF(opt)

        logger.info("开始正式训练之前，先进行一次validation:")
        validate(opt, val_loader, logger, [model_A, model_B])
        # logger.info("本次省略")

        start_epoch = 0

        logger.info("\n* Co-training")

    else:
        model_A.load_state_dict(checkpoint["model_A"])
        model_B.load_state_dict(checkpoint["model_B"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f">>>>>>>>已成功从断点处恢复训练，接下来进行epoch {start_epoch} 的训练<<<<<<<<")

    # save the history of losses from two networks
    all_loss = [[], []]
    time_point = time.time()
    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(
            "\n===============================================Epoch [{}/{}]==========================".format(epoch,
                                                                                                              opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # # Dataset split (clean, hard, noisy)
        print("--------------------本轮数据划分START--------------------")

        if resume == False and epoch == start_epoch and os.path.isfile(opt.loss_for_split):
            loss_file = torch.load(opt.loss_for_split)
            prob_A = loss_file["prob_A"]
            print("loading probability of network A, shape is: {}".format(prob_A.shape))
            prob_B = loss_file["prob_B"]
            print("loading probability of network B, shape is: {}".format(prob_B.shape))
            all_loss = loss_file["all_loss"]
            print("loading all loss, len is: {}".format(len(all_loss)))
        else:
            prob_A, prob_B, all_loss = eval_train(
                opt,
                model_A,
                model_B,
                noisy_trainloader,
                data_size,
                all_loss,
                clean_labels,
                epoch,
            )

        sc_A, c_A, n_A = split_samples_p1(prob_A, opt.p_threshold)
        sc_B, c_B, n_B = split_samples_p1(prob_B, opt.p_threshold)

        logger.info("--------观察本轮epoch，数据划分情况：")
        clean_ratio_sc_A = np.sum(clean_labels[sc_A.nonzero()[0]]) / np.sum(sc_A)
        clean_ratio_c_A = np.sum(clean_labels[c_A.nonzero()[0]]) / np.sum(c_A)
        clean_ratio_n_A = np.sum(clean_labels[n_A.nonzero()[0]]) / np.sum(n_A)

        # 在控制台输出
        # 可以保留 print 语句，以便在控制台查看信息，这样在调试时也会很方便。
        logger.info("sc_A子集有{}对数据，其中{}对是GT clean. The clean ratio is {}".format(np.sum(sc_A), np.sum(
            clean_labels[sc_A.nonzero()[0]]), clean_ratio_sc_A))
        logger.info("c_A子集有{}对数据，其中{}对是GT clean. The clean ratio is {}".format(np.sum(c_A), np.sum(
            clean_labels[c_A.nonzero()[0]]), clean_ratio_c_A))
        logger.info("n_A子集有{}对数据，其中{}对是GT clean. The clean ratio is {}".format(np.sum(n_A), np.sum(
            clean_labels[n_A.nonzero()[0]]), clean_ratio_n_A))

        clean_ratio_sc_B = np.sum(clean_labels[sc_B.nonzero()[0]]) / np.sum(sc_B)
        clean_ratio_c_B = np.sum(clean_labels[c_B.nonzero()[0]]) / np.sum(c_B)
        clean_ratio_n_B = np.sum(clean_labels[n_B.nonzero()[0]]) / np.sum(n_B)

        logger.info("sc_B子集有{}对数据，其中{}对是GT clean. The clean ratio is {}".format(np.sum(sc_B), np.sum(
            clean_labels[sc_B.nonzero()[0]]), clean_ratio_sc_B))
        logger.info("c_B子集有{}对数据，其中{}对是GT clean. The clean ratio is {}".format(np.sum(c_B), np.sum(
            clean_labels[c_B.nonzero()[0]]), clean_ratio_c_B))
        logger.info("n_B子集有{}对数据，其中{}对是GT clean. The clean ratio is {}".format(np.sum(n_B), np.sum(
            clean_labels[n_B.nonzero()[0]]), clean_ratio_n_B))

        logger.info(
            "本实验中，训练集一共有{}对数据.其中含有{}对GT clean数据".format(len(captions_train), np.sum(clean_labels)))
        logger.info("实验设置的噪声率为{}".format(opt.noise_ratio))
        logger.info("真实噪声率为{}".format(1 - (np.sum(clean_labels) / len(captions_train))))

        logger.info("--------------------本轮数据划分结束END--------------------")

        ratio_A = np.sum(sc_A) / (np.sum(sc_A) + np.sum(c_A) + np.sum(n_A))
        ratio_B = np.sum(sc_B) / (np.sum(sc_B) + np.sum(c_B) + np.sum(n_B))

        print("\nModel A training ...")
        # train model_A
        strictly_clean_data_trainloader, clean_data_trainloader, noisy_data_trainloader = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,  # 128
            opt.workers,  # 默认为1
            opt.noise_ratio,  # 0.2
            opt.noise_file,
            strictly_clean=sc_B,
            clean=c_B,
            noisy=n_B,
            prob_A=prob_A,
            prob_B=prob_B,
        )
        train_one_epoch(opt, model_A, model_B, strictly_clean_data_trainloader, clean_data_trainloader,
                        noisy_data_trainloader,
                        epoch, model="A", ratio=ratio_B, logger=logger)

        print("\nModel B training ...")
        # train model_B
        strictly_clean_data_trainloader, clean_data_trainloader, noisy_data_trainloader = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            strictly_clean=sc_A,
            clean=c_A,
            noisy=n_A,
            prob_A=prob_A,
            prob_B=prob_B,
        )
        train_one_epoch(opt, model_B, model_A, strictly_clean_data_trainloader, clean_data_trainloader,
                        noisy_data_trainloader,
                        epoch, model="B", ratio=ratio_A, logger=logger)

        print("\nValidattion ...")
        # evaluate on validation set
        rsum, bb1, bb2, bb3, bb4, bb5, bb6 = validate(opt, val_loader, logger, [model_A, model_B])

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

        # save the first stage's checkpoint
        elif epoch == opt.clean_start_epoch - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

        # save the second stage's checkpoint
        elif epoch >= opt.noisy_start_epoch - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

        # save the last checkpoint
        elif epoch == opt.num_epochs - 1:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )

        logger.info(f"本轮结束，耗时：{(time.time() - time_point) / 60}min")
        time_point = time.time()

def train_one_epoch(opt, net, net2, strictly_clean_loader, clean_loader=None, noisy_loader=None, epoch=None, model="",
                    ratio=None,logger=None):
    """
    One epoch training.
    """

    if len(strictly_clean_loader) == 0:
        print("No clean pairs! This {} epoch is skipped!".format(epoch))
        return

    strictly_clean_labels = AverageMeter("sc_labels")
    clean_labels = AverageMeter("c_labels")
    noisy_labels = AverageMeter("n_labels")
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batchTime", ":6.3f")
    data_time = AverageMeter("dataTime", ":6.3f")
    progress = ProgressMeter(
        len(strictly_clean_loader),
        [batch_time, data_time, losses, strictly_clean_labels, clean_labels, noisy_labels],
        prefix="Training Step",
    )

    # fix one network and train the other
    net.train_start()
    net2.val_start()

    if clean_loader is not None and len(clean_loader) != 0:
        clean_iter = iter(clean_loader)

    if noisy_loader is not None and len(noisy_loader) != 0:
        noisy_iter = iter(noisy_loader)

    end = time.time()

    for i, batch_train_data in enumerate(strictly_clean_loader):
        (
            batch_images_sc,
            batch_text_sc,
            batch_lengths_sc,
            sc_ids,
            batch_labels_sc,
            batch_prob_A_sc,
            batch_prob_B_sc,
            _sc_correspondence
        ) = batch_train_data


        skip_clean_train = False  # loader drop last
        skip_noisy_train = False

        if clean_loader is not None and len(clean_loader) != 0:
            try:
                (
                    batch_images_c,
                    batch_text_c,
                    batch_lengths_c,
                    c_ids,
                    batch_labels_c,
                    batch_prob_A_c,
                    batch_prob_B_c,
                    _c_correspondence
                ) = clean_iter.next()
            except:

                skip_clean_train = True

        # noisy data
        if noisy_loader is not None and len(noisy_loader) != 0:
            try:
                (
                    batch_images_n,
                    batch_text_n,
                    batch_lengths_n,
                    n_ids,
                    batch_labels_n,
                    _n_correspondence
                ) = noisy_iter.next()
            except:

                noisy_iter = iter(noisy_loader)
                (  # whether skip or not
                    batch_images_n,
                    batch_text_n,
                    batch_lengths_n,
                    n_ids,
                    batch_labels_n,
                    _n_correspondence
                ) = noisy_iter.next()

        data_time.update(time.time() - end)

        if batch_images_sc.size(0) == 1:
            break
        else:
            if batch_images_c.size(0) == 1:
                skip_clean_train = True
            if batch_images_n.size(0) == 1:
                skip_noisy_train = True

        # loading cuda
        if torch.cuda.is_available():
            batch_prob_A_sc = batch_prob_A_sc.cuda()
            batch_prob_B_sc = batch_prob_B_sc.cuda()
            batch_labels_sc = batch_labels_sc.cuda()

            batch_prob_A_c = batch_prob_A_c.cuda()
            batch_prob_B_c = batch_prob_B_c.cuda()
            batch_labels_c = batch_labels_c.cuda()

        with torch.no_grad():
            net.val_start()

            targets_sc = batch_labels_sc
            strictly_clean_labels.update(np.mean(targets_sc.cpu().numpy()), batch_images_sc.size(0))

            # --------------------clean data
            if epoch >= opt.clean_start_epoch:
                predict_clean = net.predict(batch_images_c, batch_text_c, batch_lengths_c)
                if model == "A":
                    targets_c = torch.mul(batch_prob_B_c, batch_labels_c) + torch.mul((1 - batch_prob_B_c),
                                                                                      predict_clean.t())
                elif model == "B":
                    targets_c = torch.mul(batch_prob_A_c, batch_labels_c) + torch.mul((1 - batch_prob_A_c),
                                                                                      predict_clean.t())

                clean_labels.update(np.mean(targets_c.cpu().numpy()), batch_images_c.size(0))

            # ------------------noisy data

            if epoch >= opt.noisy_start_epoch:
                targets_n, batch_text_n, batch_lengths_n = net.process_noisy_p1(batch_images_sc, batch_text_sc,
                                                                                batch_lengths_sc, batch_images_n)

                noisy_labels.update(np.mean(targets_n.cpu().numpy()), batch_images_n.size(0))

            filter=0
            filter_h=0
            filter_sc=0

        net.train_start()

        loss_sc, r_mean_sc, c_mean_sc,loss1,loss2,loss3 = net.train(
            batch_images_sc,
            batch_text_sc,
            batch_lengths_sc,
            labels=targets_sc,
            mode="train_sc",
            soft_label=opt.soft_label,
            smooth_label=opt.smooth_label,
            filter=filter_sc
        )

        if i%100==0:
            logger.info(f"调试信息:loss={loss1},loss2={loss2},loss3={loss3}")

        if epoch < opt.clean_start_epoch:
            loss_c = 0
            loss_n = 0

        elif epoch < opt.noisy_start_epoch:
            loss_c = 0
            loss_n = 0
            if skip_clean_train is not True:
                loss_c, r_mean_c, c_mean_c = net.train(
                    batch_images_c,
                    batch_text_c,
                    batch_lengths_c,
                    labels=targets_c,
                    soft_label=opt.soft_label,
                    smooth_label=opt.smooth_label,
                    mode="train",
                    filter=filter_h
                )

        else:
            loss_c = 0
            loss_n = 0

            if skip_clean_train is not True:
                loss_c, r_mean_c, c_mean_c = net.train(
                    batch_images_c,
                    batch_text_c,
                    batch_lengths_c,
                    labels=targets_c,
                    soft_label=opt.soft_label,
                    smooth_label=opt.smooth_label,
                    mode="train",
                    filter=filter_h
                )

            if skip_noisy_train is not True:
                loss_n, r_mean_n, c_mean_n = net.train(
                    batch_images_n,
                    batch_text_n,
                    batch_lengths_n,
                    labels=targets_n,
                    soft_label=opt.soft_label,
                    smooth_label=opt.smooth_label,
                    mode="train",
                    filter=filter
                )


        loss = loss_sc + loss_c + loss_n
        losses.update(loss, batch_images_sc.size(0) + batch_images_n.size(0) + batch_images_c.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % opt.log_step == 0:
            progress.display(i)


def warmup(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batchTime", ":6.3f")
    data_time = AverageMeter("dataTime", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, captions, lengths, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        loss, _, _ = model.train(images, captions, lengths, mode="warmup")
        losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, logger, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )

        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=100
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )

    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)


    logger.info(
        # 计算评估指标：R@1、R@5、R@10、中位数排名和平均排名
        "Image to text: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)



    logger.info(
        "Text to image: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    r_sum = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info(f"rsum:{r_sum}")

    return r_sum, r1, r5, r10, r1i, r5i, r10i


def eval_train(
        opt, model_A, model_B, data_loader, data_size, all_loss, clean_labels, epoch
):
    """
    Compute per-sample loss and probability
    """
    batch_time = AverageMeter("batchTime", ":6.3f")
    data_time = AverageMeter("dataTime", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )

    model_A.val_start()
    model_B.val_start()
    losses_A = torch.zeros(data_size)
    losses_B = torch.zeros(data_size)

    captions_A = [[] for i in range(data_size)]
    captions_B = [[] for i in range(data_size)]

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            loss_A = model_A.train(images, captions, lengths, mode="eval_loss")
            loss_B = model_B.train(images, captions, lengths, mode="eval_loss")
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]
                captions_A[ids[b]].append(captions[b].cpu().numpy())
                captions_B[ids[b]].append(captions[b].cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                progress.display(i)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    print("\nFitting GMM ...")

    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:,
             gmm_A.means_.argmin()]  # The probability of belonging to the first group of components (have smaller loss).

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    print("The length of computed loss of network A is: {}".format(len(input_loss_A.cpu().numpy())))
    print("The length of computed loss of network B is: {}".format(len(input_loss_B.cpu().numpy())))

    # draw = opt.draw_gmm
    draw = False  # 暂时先把画图功能关掉
    if draw:
        plot_gmm(epoch, gmm_A, input_loss_A.cpu().numpy(), clean_labels,
                 opt.output_dir + "/" + opt.data_name + '_noise_ratio_' + str(opt.noise_ratio) + '_epoch_' + str(
                     epoch) + '_gmm_A.png')
        plot_gmm(epoch, gmm_B, input_loss_B.cpu().numpy(), clean_labels,
                 opt.output_dir + "/" + opt.data_name + '_noise_ratio_' + str(opt.noise_ratio) + '_epoch_' + str(
                     epoch) + '_gmm_B.png')

    # Save the calculated loss for split dataset.
    save_loss = True
    if save_loss:
        save_loss_for_split(
            {
                "epoch": epoch,
                "prob_A": prob_A,
                "prob_B": prob_B,
                "all_loss": all_loss,
                "opt": opt,
            },
            filename="split_loss_{}.pth.tar".format(epoch),
            prefix=opt.output_dir + "/",
        )

    return prob_A, prob_B, all_loss


def split_prob(prob, threshld):
    if prob.min() > threshld:
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        # np.sort默认按从小到大顺序排序
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return pred

def split_samples_p1(prob, threshold):

    clean_prob_mask = split_prob(prob, 0.5)

    strictly_clean_mask = np.zeros(prob.shape[0], dtype=bool)
    strictly_clean_mask[prob > 0.6] = True

    clean_mask = np.zeros(prob.shape[0], dtype=bool)
    clean_mask[np.logical_and(strictly_clean_mask == False, clean_prob_mask == True)] = True

    noisy_mask = np.zeros(prob.shape[0], dtype=bool)
    noisy_mask[clean_prob_mask == False] = True

    return strictly_clean_mask, clean_mask, noisy_mask