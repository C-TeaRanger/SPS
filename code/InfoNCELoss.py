import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """
    Compute infoNCE loss
    """

    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()

        self.temperature = temperature

    def forward(
            self,
            scores,  # similarity matrix.
            labels=None,  # refined labels. when mode="predict", labels must be None.
            mode="train",
            soft_label=True,  # when true, one hot is changed to soft during calculating infoNCE loss
            smooth_label=False,  # this argument is useless when soft_label is set to False
            filter=0,  # if weight < filter, then weight = 0, only for soft_label=True, smooth_label=False
            imageMatrix=None, #为了计算loss3 额外需要的两个相似度矩阵
            textMatrix=None,
    ):

        similarity = scores.clone()  # copy a similarity as backup.
        if mode=="train_sc":

            loss2=torch.mean((similarity-similarity.T)**2)
            # loss2=loss2*15
            loss2=loss2*25

            loss3=torch.mean((imageMatrix-textMatrix)**2)
            # loss3=loss3*10
            loss3=loss3*25
            # loss3=loss3*20
            # loss3=loss3*25

            return loss3, loss2, torch.Tensor([0])

        if labels is not None:
            labels = labels.t()[0]

        scores = torch.exp(torch.div(scores, self.temperature))
        diagonal = scores.diag().view(scores.size(0), 1).t().to(scores.device)
        sum_row = scores.sum(1)
        sum_col = scores.sum(0)
        loss_text_retrieval = -torch.log(torch.div(diagonal, sum_row))
        loss_image_retrieval = -torch.log(torch.div(diagonal, sum_col))

        if mode == "predict":

            predict_text_retrieval = torch.div(scores.t(), sum_row).t()
            predict_image_retrieval = torch.div(scores, sum_col)

            p = (predict_text_retrieval + predict_image_retrieval) / 2
            p = p.diag().view(scores.size(0), 1).t().to(scores.device)
            p = p.clamp(min=0, max=1)

            return p

        elif mode == "warmup":

            return (loss_text_retrieval + loss_image_retrieval)[0].mean(), torch.Tensor([0]), torch.Tensor([0])

        elif mode == "train":

            loss = loss_text_retrieval + loss_image_retrieval
            loss = loss.to(labels.device)
            loss = torch.mul(loss, labels)

            return loss[0].mean(), torch.Tensor([0]), torch.Tensor([0])

        elif mode == "eval_loss":

            return (loss_text_retrieval + loss_image_retrieval)[0], torch.Tensor([0]), torch.Tensor([0])

