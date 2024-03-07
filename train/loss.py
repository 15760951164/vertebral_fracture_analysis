import torch
import torch.nn as nn
import math

class lambda_l2_loss(nn.Module):
    def __init__(self, lambda_l2 = 30) -> None:
        super().__init__()
        
        self.lambda_l2 = lambda_l2
        self.l2_loss = torch.nn.MSELoss()
    
    def forward(self, input, target):
        return self.lambda_l2*self.l2_loss(input, target)

class DiceL2Loss(nn.Module):

    def __init__(self, lambda_l2=20.0) -> None:
        super().__init__()

        self.lambda_l2 = lambda_l2
        self.mse = torch.nn.MSELoss()

    def dice_loss(self, input, target):
        eps = 1e-5

        input_ = input.view(-1)
        target_ = target.view(-1)

        intersec = torch.dot(input_, target_).sum()
        union = torch.sum(input_ * input_) + torch.sum(target_ * target_)

        dsc = (2.0 * intersec + eps) / (union + eps)

        return 1 - dsc

    def l2_loss(self, input, target):

        return self.mse(input, target)

    def forward(self, input, target):

        return self.dice_loss(
            input, target) + self.lambda_l2 * self.l2_loss(input, target)


class WingLoss(nn.Module):

    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class AdaptiveWingLoss(nn.Module):

    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(
            1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (
            1 /
            (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (
                self.alpha - y2) * (torch.pow(
                    self.theta / self.epsilon,
                    self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


if __name__ == "__main__":
    loss_func = DiceL2Loss()
    y = torch.ones(4, 1, 68, 64, 64)
    y_hat = torch.zeros(4, 1, 68, 64, 64)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)