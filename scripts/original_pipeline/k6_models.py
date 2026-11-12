"""PTDR K6: shared-trunk torch models for Model T (typed, factorized) and
Model M (monolithic). Identical trunk architecture for both; only the head(s)
differ. No dropout (matching this project's existing frozen small-MLP
convention in Stages 1.5/1.6, which used sklearn's MLPClassifier with no
dropout). No depth/width tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN1 = 64
HIDDEN2 = 32


def make_trunk(d_in):
    return nn.Sequential(
        nn.Linear(d_in, HIDDEN1), nn.ReLU(),
        nn.Linear(HIDDEN1, HIDDEN2), nn.ReLU(),
    )


class ModelT(nn.Module):
    """Typed, factorized: shared trunk -> act head (sigmoid) + transition head
    (softmax over the 4 non-source classes, source logit masked)."""

    def __init__(self, d_in):
        super().__init__()
        self.trunk = make_trunk(d_in)
        self.act_head = nn.Linear(HIDDEN2, 1)
        self.trans_head = nn.Linear(HIDDEN2, 5)

    def forward(self, x, source_idx):
        h = self.trunk(x)
        z_act = self.act_head(h).squeeze(-1)                 # (n,)
        z_trans = self.trans_head(h)                         # (n,5)
        mask = torch.full_like(z_trans, 0.0)
        mask.scatter_(1, source_idx.unsqueeze(1), float("-inf"))
        z_trans_masked = z_trans + mask
        log_p_trans = F.log_softmax(z_trans_masked, dim=1)   # (n,5), -inf at source col -> 0 prob
        log_p_act = F.logsigmoid(z_act)                      # log P(act=1)
        log_p_keep = F.logsigmoid(-z_act)                    # log P(act=0) = log(1-p_act)
        return log_p_act, log_p_keep, log_p_trans, z_act

    def nll_loss(self, x, source_idx, y_final):
        log_p_act, log_p_keep, log_p_trans, _ = self.forward(x, source_idx)
        is_keep = (y_final == source_idx)
        # log p_final[y] = log_p_keep                         if y == source
        #                 = log_p_act + log_p_trans[:, y]     otherwise
        log_p_trans_at_y = log_p_trans.gather(1, y_final.unsqueeze(1)).squeeze(1)
        log_p_correct = log_p_act + log_p_trans_at_y
        log_p_final_at_y = torch.where(is_keep, log_p_keep, log_p_correct)
        return -log_p_final_at_y.mean()

    def predict_p_final(self, x, source_idx):
        with torch.no_grad():
            log_p_act, log_p_keep, log_p_trans, z_act = self.forward(x, source_idx)
            p_act = torch.sigmoid(z_act)
            p_trans = torch.exp(log_p_trans)
            n = x.shape[0]
            p_final = p_act.unsqueeze(1) * p_trans
            p_final.scatter_(1, source_idx.unsqueeze(1), 0.0)
            keep_val = torch.exp(log_p_keep)
            p_final.scatter_(1, source_idx.unsqueeze(1), keep_val.unsqueeze(1))
            return p_final, p_act, p_trans


class ModelM(nn.Module):
    """Monolithic: shared trunk -> single 5-way softmax head. No auxiliary gate,
    no transition head, no source masking (source class is present only as an
    input feature, via S2's one-hot block, same as Model T)."""

    def __init__(self, d_in):
        super().__init__()
        self.trunk = make_trunk(d_in)
        self.final_head = nn.Linear(HIDDEN2, 5)

    def forward(self, x):
        h = self.trunk(x)
        z_final = self.final_head(h)
        return z_final

    def nll_loss(self, x, y_final):
        z_final = self.forward(x)
        log_p_final = F.log_softmax(z_final, dim=1)
        return F.nll_loss(log_p_final, y_final)

    def predict_p_final(self, x):
        with torch.no_grad():
            z_final = self.forward(x)
            p_final = F.softmax(z_final, dim=1)
            return p_final


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
