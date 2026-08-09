"""诊断脚本：分阶段测量 MLP_label 训练管线的 wall 时间与 CPU 时间.

用法: uv run python LinearProbe_evaluation/diagnose_cpu.py

输出每个阶段的 wall 耗时、进程 CPU 时间（time.process_time）与 CPU 占用率
（CPU 时间 / wall 时间；>1 表示多核并行）。同时校验训练全流程 GPU 参与
（模型参数与每个 batch 均在 cuda 上）。
"""
import sys
import threading
import time
from pathlib import Path

# 确保项目根目录在 path 中（直接脚本运行时）：移除脚本目录，避免本地
# qdrant_client.py 遮蔽 pip 包导致循环导入（同 webui.py 的处理）。
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path = [p for p in sys.path if Path(p).resolve() != _SCRIPT_DIR]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

import torch

from LinearProbe_evaluation.qdrant_client import QdrantManager
from LinearProbe_evaluation.dataset import stratified_train_val_split
from LinearProbe_evaluation.trainer import TrainConfig, train_mlp


def _report(stage: str, wall_start: float, cpu_start: float) -> tuple[float, float]:
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    util = cpu / wall if wall > 0 else 0.0
    print(
        f"[{stage}] wall={wall:6.2f}s | cpu={cpu:6.2f}s | "
        f"CPU占用率={util:5.2f} 核"
    )
    return wall, cpu


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA 不可用，无法诊断", file=sys.stderr)
        return 1

    manager = QdrantManager()
    if not manager.health_check():
        print("Qdrant 不可达", file=sys.stderr)
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__} | OMP线程数: {torch.get_num_threads()}")

    # ---- 阶段 0：加载采样地图（480MB JSON） ----
    w0, c0 = time.perf_counter(), time.process_time()
    from KNN_evaluation.sampling_map import ensure_sampling_map
    smap = ensure_sampling_map(manager)
    _report("阶段0 采样地图加载", w0, c0)
    print(f"      地图 ID 总数: {sum(len(v) for v in (smap.get('by_label') or {}).values()):,}")

    # ---- 阶段 1：分层采样（下载样本量） ----
    w1, c1 = time.perf_counter(), time.process_time()
    train_ds, val_ds = stratified_train_val_split(
        manager, train_per_class=2000, val_per_class=500, seed=42,
    )
    _report("阶段1 分层采样", w1, c1)
    print(f"      训练集 {train_ds.size:,} | 验证集 {val_ds.size:,}")

    # ---- 阶段 2：训练（纯 GPU） ----
    w2, c2 = time.perf_counter(), time.process_time()
    config = TrainConfig(epochs=10, batch_size=2048, lr=1e-2, device="cuda",
                         output_dir="outputs/diag_cpu")
    gpu_times: list[float] = []

    def cb(event: dict) -> None:
        if event["event"] == "epoch_end":
            gpu_times.append(event["train_loss"])

    result = train_mlp(train_ds, val_ds, config, progress_callback=cb)
    _report("阶段2 CUDA训练", w2, c2)
    print(f"      设备: {result.device_name} | GPU kernel: {result.gpu_kernel_seconds:.2f}s")

    # ---- 阶段 3：收尾报告（best checkpoint 推理） ----
    w3, c3 = time.perf_counter(), time.process_time()
    from LinearProbe_evaluation.trainer import load_mlp_label, predict_labels
    best_model, _ = load_mlp_label(result.best_checkpoint, "cuda")
    pred = predict_labels(best_model, val_ds.X, "cuda")
    _report("阶段3 收尾报告", w3, c3)
    print(f"      验证集预测完成: {pred.shape[0]} 样本")

    print(f"\n总耗时 {time.perf_counter() - w0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
