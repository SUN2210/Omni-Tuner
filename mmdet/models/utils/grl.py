import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd  # 在上下文中缓存系数，供反向传播使用
        return x.view_as(x)  # 前向不做任何修改，原样返回输入

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None  # 反向时乘以 -λ，实现梯度反转


def grad_reverse(x, lambd=1.0):
    return GradientReversalFunction.apply(x, lambd)  # Function 调用入口，便于复用


class GradientReversal(torch.nn.Module):
    """Layer that applies the gradient reversal trick."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd  # 保存反转系数 λ，控制对抗强度

    def forward(self, x):
        return grad_reverse(x, self.lambd)  # 在前向中调用自定义 Function
