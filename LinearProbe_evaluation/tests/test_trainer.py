"""Tests for MLP_label trainer."""
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from LinearProbe_evaluation.trainer import (
    TrainConfig, TrainResult, train_mlp, resolve_device, load_mlp_label,
    predict_labels, CancelledError,
)
from LinearProbe_evaluation.model import MLPLabel
from LinearProbe_evaluation.tests.helpers import synthetic_pixel_dataset


@pytest.fixture
def small_ds():
    """每类 8 个样本（共 72），噪声较小保证 loss 下降可观察."""
    return synthetic_pixel_dataset(n_per_class=8, seed=42, noise=0.3)


def test_resolve_device():
    """只支持 cuda：cuda/auto 都解析为 cuda，cpu 被拒绝（禁用 CPU fallback）."""
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") == "cuda"
    assert resolve_device(None) == "cuda"
    with pytest.raises(ValueError):
        resolve_device("cpu")


def test_resolve_device_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("LinearProbe_evaluation.trainer.torch.cuda.is_available",
                        lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_train_runs_and_improves(small_ds, tmp_path):
    """小规模训练：完成、history 结构完整、best/final checkpoint 与 meta 落盘."""
    train_ds = small_ds
    val_ds = small_ds
    config = TrainConfig(
        epochs=5, batch_size=16, lr=1e-2, device="cuda", seed=42,
        output_dir=tmp_path / "out",
    )
    result = train_mlp(train_ds, val_ds, config)

    assert isinstance(result, TrainResult)
    assert len(result.history) == 5
    keys = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1"}
    assert keys <= set(result.history[0])
    assert 0 <= result.best_val_accuracy <= 1
    assert result.best_epoch >= 1

    # 训练应使 loss 下降（线性可分合成数据）
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert result.history[-1]["train_acc"] >= result.history[0]["train_acc"]

    # checkpoint 与 meta 文件存在
    assert result.best_checkpoint is not None and result.best_checkpoint.exists()
    assert result.final_checkpoint is not None and result.final_checkpoint.exists()
    meta_path = tmp_path / "out" / "mlp_label_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["architecture"]["class"] == "MLPLabel"
    assert len(meta["history"]) == 5

    # 分类报告结构
    assert "accuracy" in result.val_report
    assert "per_class" in result.val_report


def test_checkpoint_roundtrip(small_ds, tmp_path):
    """保存 → load_mlp_label 加载后前向结果一致."""
    config = TrainConfig(epochs=2, batch_size=16, lr=1e-2, device="cuda",
                         output_dir=tmp_path / "out")
    result = train_mlp(small_ds, small_ds, config)
    model, meta = load_mlp_label(result.best_checkpoint, "cpu")
    assert isinstance(model, MLPLabel)
    assert meta["architecture"]["in_features"] == 64
    assert meta["architecture"]["num_classes"] == 9
    assert tuple(meta["architecture"]["hidden_dims"]) == (256, 256, 256)
    x = torch.randn(4, 64)
    assert model(x).shape == (4, 9)


def test_predict_labels_shape(small_ds):
    model = MLPLabel()
    pred = predict_labels(model, small_ds.X[:10], "cpu")
    assert pred.shape == (10,)
    assert set(np.unique(pred)) <= set(range(9))


def test_empty_train_raises():
    from LinearProbe_evaluation.dataset import empty_dataset
    with pytest.raises(ValueError, match="训练集为空"):
        train_mlp(empty_dataset(), empty_dataset(), TrainConfig(epochs=1))


def test_cancel_event(small_ds, tmp_path):
    """取消事件预先置位 → 抛出 CancelledError."""
    ev = threading.Event()
    ev.set()
    with pytest.raises(CancelledError):
        train_mlp(small_ds, small_ds, TrainConfig(epochs=2, output_dir=tmp_path / "o"),
                  cancel_event=ev)


def test_progress_callback_events(small_ds, tmp_path):
    """进度回调事件序列：epoch_start → batch* → epoch_end 循环."""
    events: list[str] = []
    epochs_seen: set[int] = set()
    config = TrainConfig(epochs=2, batch_size=16, lr=1e-2, device="cuda",
                         log_interval=1, output_dir=tmp_path / "o")

    def cb(event):
        events.append(event["event"])
        if event["event"] == "epoch_end":
            epochs_seen.add(event["epoch"])

    train_mlp(small_ds, small_ds, config, progress_callback=cb)
    assert events[0] == "epoch_start"
    assert "batch" in events
    assert events[-1] == "epoch_end"
    assert epochs_seen == {1, 2}


def test_sgd_optimizer(small_ds, tmp_path):
    config = TrainConfig(epochs=2, batch_size=16, lr=1e-2, optimizer="sgd",
                         device="cuda", output_dir=tmp_path / "o")
    result = train_mlp(small_ds, small_ds, config)
    assert result.best_val_accuracy >= 0


def test_invalid_optimizer(small_ds, tmp_path):
    with pytest.raises(ValueError, match="optimizer"):
        train_mlp(small_ds, small_ds,
                  TrainConfig(epochs=1, optimizer="nope", output_dir=tmp_path / "o"))


def test_no_checkpoint_when_disabled(small_ds, tmp_path):
    """checkpoint_best=False + save_final=False：不落盘、best 回填为最终指标."""
    config = TrainConfig(epochs=2, batch_size=16, lr=1e-2, device="cuda",
                         output_dir=tmp_path / "o",
                         checkpoint_best=False, save_final=False)
    result = train_mlp(small_ds, small_ds, config)
    assert result.best_checkpoint is None
    assert result.final_checkpoint is None
    assert result.best_epoch == 2  # 回填为最后一个 epoch
    assert result.best_val_accuracy == result.history[-1]["val_acc"]


def test_linear_variant_training(small_ds, tmp_path):
    """model_variant='linear' 训练跑通；checkpoint 记录 hidden_dims=[] 并正确重建."""
    config = TrainConfig(epochs=2, batch_size=16, lr=1e-2, device="cuda",
                         model_variant="linear", output_dir=tmp_path / "o")
    result = train_mlp(small_ds, small_ds, config)
    assert result.best_val_accuracy >= 0
    model, meta = load_mlp_label(result.best_checkpoint, "cpu")
    assert model.variant == "linear"
    assert meta["architecture"]["hidden_dims"] == []
    x = torch.randn(4, 64)
    assert model(x).shape == (4, 9)


def test_invalid_model_variant(small_ds, tmp_path):
    with pytest.raises(ValueError, match="模型结构"):
        train_mlp(small_ds, small_ds,
                  TrainConfig(epochs=1, model_variant="nope",
                              output_dir=tmp_path / "o"))


def test_empty_val_set_no_nan(small_ds, tmp_path):
    """空验证集（val-ratio 0）：val 指标记 None、不参与 best、meta JSON 可解析."""
    from LinearProbe_evaluation.dataset import empty_dataset
    config = TrainConfig(epochs=3, batch_size=16, lr=1e-2, device="cuda",
                         output_dir=tmp_path / "o")
    result = train_mlp(small_ds, empty_dataset(), config)
    # 空 val 时 best 回填为最后一个 epoch（不把 epoch 1 当 best）
    assert result.best_epoch == 3
    for rec in result.history:
        assert rec["val_loss"] is None
        assert rec["val_acc"] is None
    # meta JSON 可被 json.loads 解析（无 NaN）
    meta_path = tmp_path / "o" / "mlp_label_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta["history"]) == 3
    assert meta["history"][0]["val_loss"] is None


def test_empty_val_with_progress_callback(small_ds, tmp_path):
    """空验证集 + 进度回调：epoch_end 事件 val_* 为 None，消费端格式化不崩溃."""
    from LinearProbe_evaluation.dataset import empty_dataset
    events: list[dict] = []
    config = TrainConfig(epochs=2, batch_size=16, lr=1e-2, device="cuda",
                         output_dir=tmp_path / "o")

    def cb(event):
        events.append(event)
        # 模拟 CLI/WebUI 消费端：None 需占位为 0.0 后再格式化
        if event["event"] == "epoch_end":
            val_acc = event["val_acc"] if event["val_acc"] is not None else 0.0
            assert f"{val_acc:.4f}" is not None
            assert event["val_loss"] is None

    train_mlp(small_ds, empty_dataset(), config, progress_callback=cb)
    epoch_ends = [e for e in events if e["event"] == "epoch_end"]
    assert len(epoch_ends) == 2
    assert all(e["val_acc"] is None for e in epoch_ends)
