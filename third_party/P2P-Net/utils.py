import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)









def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets):
    if torch.cuda.is_available():
        loss = Variable(torch.zeros(1).cuda())
    else:
        loss = Variable(torch.zeros(1))
    batch_size = score.size(0)

    if torch.cuda.is_available():
        data_type = torch.cuda.FloatTensor
    else:
        data_type = torch.FloatTensor
    for i in range(targets.shape[1]):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(data_type)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


def smooth_CE(logits, label, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1-peak)/(num_cls-1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1)) # batch average

    return loss
