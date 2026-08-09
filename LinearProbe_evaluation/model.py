"""MLP_label 模型定义 — 64 维输入 → 9 维输出，支持两种结构（web 端可选）.

当前 Qdrant 中保存的是像素 DW 数据集的 **硬分类 label**（0-8），因此
``MLPLabel`` 输出 9 维 logits，对应 9 个标签的硬分类独热码语义
（推理时 ``argmax`` 得到类别）。

**两种结构（``hidden_dims`` 决定，web 端 / CLI 可选）：**

1. **linear**（标准 Linear Probe，``hidden_dims=()``，585 参数）::

       Linear(64, 9)

   学术惯例（Alain & Bengio 2016 等）中的 linear probe 定义：单一线性层、
   无隐藏层、无激活，用于检测 embedding 的**线性可分性**。

2. **mlp**（LeNet-5 规模多层 MLP，``hidden_dims=(256,256,256)``，约 15 万参数）::

       Linear(64, 256) → ReLU → Dropout(0.2)
       Linear(256, 256) → ReLU → Dropout(0.2)
       Linear(256, 256) → ReLU → Dropout(0.2)
       Linear(256, 9)

   输入为 64 维像素 embedding（1 维向量而非图像），因此采用全连接 MLP 而非 CNN；
   隐含层维度可经 ``hidden_dims`` 自定义。

后续若引入像素 DW 数据集的 **prob 信息**（软分类概率），将新增
``MLPProb``（同为 64→9，但输出为概率分布，用软标签/回归目标训练），
本类保持硬分类语义不变。
"""
import torch.nn as nn

from LinearProbe_evaluation.config import VECTOR_SIZE, NUM_CLASSES

# 默认隐含层维度：对标 LeNet-5 规模（参数量 ≈ 15 万）
DEFAULT_HIDDEN_DIMS: tuple[int, ...] = (256, 256, 256)
DEFAULT_DROPOUT: float = 0.2

# 两种预置结构 variant → hidden_dims 映射
VARIANT_LINEAR = "linear"
VARIANT_MLP = "mlp"
VARIANT_HIDDEN_DIMS: dict[str, tuple[int, ...]] = {
    VARIANT_LINEAR: (),            # 单层 Linear(64, 9)
    VARIANT_MLP: DEFAULT_HIDDEN_DIMS,
}


class MLPLabel(nn.Module):
    """MLP_label：64 维像素 embedding → 9 类硬标签的分类 MLP.

    ``hidden_dims=()`` 时为标准 Linear Probe（单层）；非空时为多层 MLP
    （ReLU + Dropout 交替，默认 LeNet-5 规模）。

    Attributes:
        in_features: 输入维度（像素 embedding 维度，默认 64）.
        num_classes: 输出类别数（DW 标签数，默认 9）.
        hidden_dims: 隐含层维度序列（() = 单层线性）.
        dropout: 隐含层 Dropout 概率（默认 0.2）.
    """

    def __init__(
        self,
        in_features: int = VECTOR_SIZE,
        num_classes: int = NUM_CLASSES,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        # None → 默认 LeNet-5 规模多层；显式 () → 单层 Linear Probe
        self.hidden_dims = tuple(
            hidden_dims if hidden_dims is not None else DEFAULT_HIDDEN_DIMS
        )
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev = in_features
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    @property
    def variant(self) -> str:
        """结构类型：'linear'（单层 Linear Probe）或 'mlp'（多层 MLP）."""
        return VARIANT_LINEAR if not self.hidden_dims else VARIANT_MLP

    def forward(self, x):
        """前向：x (B, in_features) → logits (B, num_classes)."""
        return self.net(x)

    def describe(self) -> str:
        """单行结构描述（用于日志/UI 展示）."""
        n_params = sum(p.numel() for p in self.parameters())
        if not self.hidden_dims:
            return (
                f"MLPLabel({self.in_features} → {self.num_classes}, "
                f"{n_params:,} 参数, Linear Probe 单层)"
            )
        hidden = " → ".join(str(h) for h in self.hidden_dims)
        return (
            f"MLPLabel({self.in_features} → {hidden} → {self.num_classes}, "
            f"ReLU+Dropout({self.dropout}), {n_params:,} 参数, LeNet-5 规模)"
        )


def build_model(
    variant: str = VARIANT_MLP,
    in_features: int = VECTOR_SIZE,
    num_classes: int = NUM_CLASSES,
    dropout: float = DEFAULT_DROPOUT,
) -> MLPLabel:
    """按 variant 构建模型：'linear' = 单层 Linear Probe，'mlp' = 多层 MLP.

    Raises:
        ValueError: 未知 variant.
    """
    if variant not in VARIANT_HIDDEN_DIMS:
        raise ValueError(
            f"未知模型结构 {variant!r}，可选: {list(VARIANT_HIDDEN_DIMS)}"
        )
    return MLPLabel(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dims=VARIANT_HIDDEN_DIMS[variant],
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    """模型可训练参数量."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
