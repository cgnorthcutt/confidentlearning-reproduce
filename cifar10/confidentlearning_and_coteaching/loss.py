import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions
def loss_coteaching(
        y_1,
        y_2,
        t,
        forget_rate,
        class_weights=None,
):
    """Co-Teaching Loss function.
    Source: https://github.com/bhanML/Co-teaching/blob/master/loss.py

    Parameters
    ----------
    y_1 : Tensor array
      Output logits from model 1

    y_2 : Tensor array
      Output logits from model 2

    t : np.array
      List of Noisy Labels (t standards for targets)

    forget_rate : float
      Decimal between 0 and 1 for how quickly the models forget what they learn.
      Just use rate_schedule[epoch] for this value

    ind : iterable / list / np.array
      Indices from train_loader created like this:
      for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()

    noise_or_not : np.array
      Np.array of length number of total training examples in dataset that says
      for every training example if its prediction is equal to its given label.
      For each example in the training dataset, true if equal, else false.

    class_weights : Tensor array, shape (Number of classes x 1), Default: None
      A np.torch.tensor list of length number of classes with weights
    """

    loss_1 = F.cross_entropy(y_1, t, reduce = False, weight=class_weights)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False, weight=class_weights)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], weight=class_weights)
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], weight=class_weights)

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2


