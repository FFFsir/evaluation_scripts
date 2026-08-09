"""Tests for MLPLabel model definition (LeNet-5 规模多层 MLP)."""
import pytest
import torch

from LinearProbe_evaluation.model import MLPLabel, DEFAULT_HIDDEN_DIMS
from LinearProbe_evaluation.config import VECTOR_SIZE, NUM_CLASSES


def test_forward_shape():
    """输入 (B, 64) → 输出 (B, 9)."""
    model = MLPLabel()
    x = torch.randn(16, VECTOR_SIZE)
    out = model(x)
    assert out.shape == (16, NUM_CLASSES)


def test_multi_layer_structure():
    """LeNet-5 规模：多层全连接（隐含层 ≥2 层 + ReLU + Dropout），非单层线性."""
    model = MLPLabel()
    assert model.hidden_dims == DEFAULT_HIDDEN_DIMS
    # net 序列：每隐含层 3 个子模块（Linear/ReLU/Dropout）+ 输出 Linear
    expected_len = len(DEFAULT_HIDDEN_DIMS) * 3 + 1
    assert len(model.net) == expected_len
    # 第一层是 Linear(64, 256)
    assert isinstance(model.net[0], torch.nn.Linear)
    assert model.net[0].in_features == VECTOR_SIZE
    assert model.net[0].out_features == DEFAULT_HIDDEN_DIMS[0]
    # 隐含层之间夹 ReLU 与 Dropout
    assert isinstance(model.net[1], torch.nn.ReLU)
    assert isinstance(model.net[2], torch.nn.Dropout)
    # 输出层 Linear(256, 9)
    out_linear = model.net[-1]
    assert isinstance(out_linear, torch.nn.Linear)
    assert out_linear.in_features == DEFAULT_HIDDEN_DIMS[-1]
    assert out_linear.out_features == NUM_CLASSES


def test_parameter_scale_comparable_to_lenet5():
    """参数量对标 LeNet-5 量级（≥ 10 万），不是单层线性（仅 ~600 参数）."""
    model = MLPLabel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params >= 100_000, f"参数量 {n_params} 过小，未达到 LeNet-5 规模"
    # 64*256 + 256*256 + 256*256 + 256*9 偏置等 → 精确值
    expected = (
        VECTOR_SIZE * DEFAULT_HIDDEN_DIMS[0] + DEFAULT_HIDDEN_DIMS[0]
        + DEFAULT_HIDDEN_DIMS[0] * DEFAULT_HIDDEN_DIMS[1] + DEFAULT_HIDDEN_DIMS[1]
        + DEFAULT_HIDDEN_DIMS[1] * DEFAULT_HIDDEN_DIMS[2] + DEFAULT_HIDDEN_DIMS[2]
        + DEFAULT_HIDDEN_DIMS[2] * NUM_CLASSES + NUM_CLASSES
    )
    assert n_params == expected


def test_linear_variant_single_layer():
    """hidden_dims=() → 标准 Linear Probe：单一 Linear(64,9)，585 参数."""
    model = MLPLabel(hidden_dims=())
    assert model.variant == "linear"
    assert len(model.net) == 1
    assert isinstance(model.net[0], torch.nn.Linear)
    assert model.net[0].in_features == VECTOR_SIZE
    assert model.net[0].out_features == NUM_CLASSES
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == VECTOR_SIZE * NUM_CLASSES + NUM_CLASSES  # 585
    x = torch.randn(4, VECTOR_SIZE)
    assert model(x).shape == (4, NUM_CLASSES)


def test_build_model_variants():
    """build_model：mlp / linear / 未知 variant."""
    from LinearProbe_evaluation.model import build_model, VARIANT_LINEAR, VARIANT_MLP
    m_mlp = build_model(VARIANT_MLP)
    m_lin = build_model(VARIANT_LINEAR)
    assert m_mlp.variant == "mlp" and m_lin.variant == "linear"
    assert sum(p.numel() for p in m_mlp.parameters()) >= 100_000
    assert sum(p.numel() for p in m_lin.parameters()) == 585
    with pytest.raises(ValueError):
        build_model("nope")


def test_custom_dimensions():
    """自定义输入/输出维度与隐含层."""
    model = MLPLabel(in_features=128, num_classes=5, hidden_dims=(64, 32), dropout=0.1)
    assert model(torch.randn(2, 128)).shape == (2, 5)
    assert model.hidden_dims == (64, 32)
    assert model.dropout == 0.1


def test_describe():
    model = MLPLabel()
    s = model.describe()
    assert "MLPLabel" in s and "64" in s and "9" in s and "LeNet-5" in s
    assert "参数" in s


def test_label_names_mapping():
    """9 类标签映射与 DW 数据集一致（0-8）."""
    from LinearProbe_evaluation.label_mapping import LABEL_NAMES
    assert len(LABEL_NAMES) == 9
    assert LABEL_NAMES[0] == "water 水"
    assert LABEL_NAMES[8] == "snow_and_ice 冰雪"
