
import torch
import torch.nn.functional as F


class DeepPoly:
    def __init__(self, size, lb, ub):
        iden = torch.diag(torch.ones(size))
        self.slb = torch.cat([iden, torch.zeros(size).unsqueeze(1)], dim=1)
        self.sub = self.slb
        self.lb = lb
        self.ub = ub
        self.hist = []
        self.layers = 0
        self.is_relu = False

    def save(self):
        """ Save all constraints for the back substitution """

        # append bias as last row with a one s.t. matmul in backsub works correctly
        lb = torch.cat([self.lb, torch.ones(1)])
        ub = torch.cat([self.ub, torch.ones(1)])
        if not self.is_relu:
            keep_bias = torch.zeros(1, self.slb.shape[1])
            keep_bias[0, self.slb.shape[1] - 1] = 1
            slb = torch.cat([self.slb, keep_bias], dim=0)
            sub = torch.cat([self.sub, keep_bias], dim=0)
        else:
            slb = self.slb
            sub = self.sub
        self.layers += 1
        self.hist.append((slb, sub, lb, ub, self.is_relu))

        return self

    """
        Resolves the given constraints using the lower and upper bounds
        of the history at layer 'layer' to new constraints.
        lower determines if we search for lower or upper bounds
    """
    def resolve(self, cstr, layer, lower=True):
        cstr_relu_pos = F.relu(cstr)
        cstr_relu_neg = F.relu(-cstr)
        dp = self.hist[layer]
        is_relu = dp[-1]
        if layer == 0:
            lb, ub = dp[2], dp[3]
        else:
            lb, ub = dp[0], dp[1]
        if not lower:  # switch lb and ub
            lb, ub = ub, lb
        if is_relu:
            lb_diag, lb_bias = lb[0], lb[1]
            ub_diag, ub_bias = ub[0], ub[1]
            lb_bias = torch.cat([lb_bias, torch.ones(1)])
            ub_bias = torch.cat([ub_bias, torch.ones(1)])
            # the matrix multiplication with a diagonal matrix is nothing
            # but a column wise scaling 
            # -- the bias is a matrix x vector mul
            m1 = torch.cat([cstr_relu_pos[:, :-1] * lb_diag, torch.matmul(cstr_relu_pos, lb_bias).unsqueeze(1)], dim=1)
            m2 = torch.cat([cstr_relu_neg[:, :-1] * ub_diag, torch.matmul(cstr_relu_neg, ub_bias).unsqueeze(1)], dim=1)
            return m1 - m2
        else:
            return torch.matmul(cstr_relu_pos, lb) - torch.matmul(cstr_relu_neg, ub)
