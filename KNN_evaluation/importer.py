"""批量导入管线：将 SE/DW 像素数据批量导入 Qdrant."""
import time
import uuid
import warnings
from pathlib import Path
from typing import Callable
import numpy as np
from qdrant_client import models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http import exceptions as qdrant_http_exceptions

from KNN_evaluation.config import BATCH_SIZE, VECTOR_SIZE
from KNN_evaluation.label_mapping import LABEL_NAMES
from KNN_evaluation.data_loader import PixelDataLoader, ImagePair
from KNN_evaluation.coordinate_utils import compute_utm_grid, compute_utm_grid_from_name
from KNN_evaluation.manifest import update_manifest


def _is_transient_error(exc: BaseException) -> bool:
    """判断异常是否属于可重试的瞬时失败.

    只对瞬时故障（网络抖动、超时、服务端繁忙）重试，不吞掉持久性错误：

    - `ResponseHandlingException`：底层网络/传输层异常（连接重置、超时、DNS 失败等）——
      属典型的瞬时失败。
    - `UnexpectedResponse`：HTTP 层非 2xx 响应。429（服务端繁忙/限流）与 5xx（服务端错误）
      视为瞬时失败；其余 4xx（如 400 参数错误）为持久性错误，直接抛出。
    - `ResourceExhaustedResponse`：Qdrant 429 限流且带 Retry-After 时抛出的专用异常
      （独立于 UnexpectedResponse，无 status_code 属性），瞬时失败。
    """
    if isinstance(exc, qdrant_http_exceptions.ResponseHandlingException):
        return True
    if isinstance(exc, qdrant_http_exceptions.UnexpectedResponse):
        return exc.status_code is not None and (exc.status_code == 429 or exc.status_code >= 500)
    if isinstance(exc, ResourceExhaustedResponse):
        return True
    return False


def _retry_call(fn, *args, retries: int = 3, base_delay: float = 1.0, **kwargs):
    """以指数退避重试调用 ``fn(*args, **kwargs)``，应对 Qdrant 瞬时失败.

    策略：``delay = base_delay * 2**attempt``（如 1s → 2s → 4s），每次失败后
    sleep 再重试，共尝试 ``1 + retries`` 次；达到重试上限仍失败时抛出最后一个异常.

    Args:
        fn: 被调用的可调用对象.
        *args: 传给 fn 的位置参数.
        retries: 失败后的重试次数（默认 3，即最多尝试 1 + 3 = 4 次）.
        base_delay: 首次重试前的基础延迟秒数（默认 1.0）.
        **kwargs: 传给 fn 的关键字参数.

    Returns:
        fn 的返回值（首次成功或某次重试成功）.

    Raises:
        持久性错误不重试，立即抛出；瞬时错误达重试上限后抛出最后一个异常.
    """
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — 按瞬时/持久分类决定是否重试
            if attempt >= retries or not _is_transient_error(exc):
                raise
            time.sleep(base_delay * 2**attempt)


class PixelImporter:
    """像素数据批量导入器.

    编排 scan → load → compute coords → build points → upsert 流程.
    """

    def __init__(self, manager, batch_size: int = BATCH_SIZE):
        self.manager = manager
        self.batch_size = batch_size
        self._stats: dict = {}

    def build_points(
        self,
        se_data: np.ndarray,
        dw_data: np.ndarray,
        easting: np.ndarray,
        northing: np.ndarray,
        utm_zone: int | None,
        image_id: str,
    ) -> list[models.PointStruct]:
        """将单张影像的所有像素构造为 Qdrant PointStruct 列表（向量化）.

        Args:
            se_data: (64, 128, 128) float64 embedding 数据.
            dw_data: (128, 128) uint8 标签矩阵.
            easting: (128, 128) float64 UTM 东向坐标.
            northing: (128, 128) float64 UTM 北向坐标.
            utm_zone: UTM 投影带号，None 时记为 -1.
            image_id: 影像标识.

        Returns:
            长度为 16384 的 PointStruct 列表，点序为 row-major
            （row 外层、col 内层），与逐像素循环版本完全一致.
        """
        zone = utm_zone if utm_zone is not None else -1

        # 用 image_id 作为 UUID 命名空间，保证同 image_id 下的
        # row/col 组合产生唯一且确定的 UUID
        ns = uuid.uuid5(uuid.NAMESPACE_DNS, image_id)

        # 向量化构造：(64, 128, 128) -> (16384, 64)，row-major 展平
        # .tolist() 提供 Qdrant 序列化所需的 Python 原生 float
        vectors = se_data.reshape(64, -1).T.tolist()

        # 标签与 label_name 一次展平
        # payload "label" 用 int(labels[i])：与逐像素旧循环的 int(dw_data[row,col])
        # 及 label_names 的 int cast 防御性对齐，保证存入库的标签恒为 Python int
        labels = [int(l) for l in dw_data.reshape(-1).tolist()]
        label_names = [LABEL_NAMES.get(l, "unknown") for l in labels]

        # UTM 坐标展平（row-major）
        eastings = [float(e) for e in easting.reshape(-1)]
        northings = [float(n) for n in northing.reshape(-1)]

        # 行列网格：indexing="ij" 使 rows[i,j]=i, cols[i,j]=j，row-major 展平
        rows_grid, cols_grid = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
        rows = rows_grid.reshape(-1).tolist()
        cols = cols_grid.reshape(-1).tolist()

        return [
            models.PointStruct(
                id=str(uuid.uuid5(ns, f"{r}_{c}")),
                vector=vectors[i],
                payload={
                    "label": labels[i],
                    "label_name": label_names[i],
                    "utm_easting": eastings[i],
                    "utm_northing": northings[i],
                    "utm_zone": zone,
                    "image_id": image_id,
                    "pixel_row": r,
                    "pixel_col": c,
                },
            )
            for i, (r, c) in enumerate(zip(rows, cols))
        ]

    def _batch_upsert(
        self,
        points: list[models.PointStruct],
        batch_callback: Callable[[int], None] | None = None,
    ) -> None:
        """分批 upsert points 到 Qdrant.

        Args:
            points: PointStruct 列表.
            batch_callback: 每批 upsert 成功后回调该批点数（内部使用，用于进度推进）.
        """
        for i in range(0, len(points), self.batch_size):
            batch = points[i:i + self.batch_size]
            _retry_call(
                self.manager.client.upsert,
                collection_name=self.manager.collection_name,
                points=batch,
                wait=True,
            )
            if batch_callback is not None:
                batch_callback(len(batch))

    def import_image_pair(
        self,
        pair: ImagePair,
        _batch_callback: Callable[[int], None] | None = None,
    ) -> tuple[int, int]:
        """导入一对 SE + DW 文件的所有像素.

        Args:
            pair: 匹配的 ImagePair.
            _batch_callback: 可选内部回调，每批 upsert 成功后调用（私有，用于进度线程）.

        Returns:
            (imported, skipped) — 新导入的点数和跳过的点数.
        """
        # 断点续传检查
        existing_count = _retry_call(PixelDataLoader.check_image_count, pair.image_id, self.manager)
        total = 128 * 128

        if existing_count >= total:
            return 0, total

        if existing_count > 0:
            warnings.warn(
                f"{pair.image_id}: 已存在 {existing_count} 条记录（共 {total}），将覆盖重传"
            )

        # 加载数据
        se_data = PixelDataLoader.load_se(pair.se_path)
        dw_data = PixelDataLoader.load_dw(pair.dw_path)

        # UTM 坐标：优先从文件名坐标段推算（不加载 TIF）
        try:
            lon, lat = PixelDataLoader.parse_location_coord(pair.image_id)
            easting, northing, utm_zone = compute_utm_grid_from_name(lon, lat)
            derived_from_name = True
        except (ValueError, ImportError):
            # 仅捕获可预期的解析/依赖错误并回退 GeoTIFF；真正的实现 bug
            # 不应被静默掩盖为"回退"
            warnings.warn(f"{pair.image_id}: 文件名坐标推算失败，回退 GeoTIFF")
            easting, northing, utm_zone = compute_utm_grid(pair.tif_path)
            derived_from_name = False

        if pair.tif_path is None:
            if derived_from_name:
                warnings.warn(f"{pair.image_id}: 未找到 GeoTIFF，已使用文件名坐标推算 UTM")
            else:
                warnings.warn(f"{pair.image_id}: 未找到 GeoTIFF，UTM 坐标设为 NaN")

        # 构建并导入
        points = self.build_points(se_data, dw_data, easting, northing, utm_zone, pair.image_id)
        self._batch_upsert(points, _batch_callback)

        return total, existing_count

    def import_directory(
        self,
        data_dir: Path,
        no_resume: bool = False,
        reindex: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """完整导入流程：扫描目录 → 配对 → 逐文件导入.

        Args:
            data_dir: 数据根目录（含 SE/ 和 DW/ 子目录）.
            no_resume: True 时不检查断点续传，强制重新导入.
            reindex: True 时导入完成后触发 HNSW 全量索引重建.
            progress_callback: 可选进度回调 (imported_so_far, total)，
                每批 upsert 成功后调用；默认 None 时行为与现状完全一致.

        Returns:
            统计字典:
            {
                "total_pixels": int,
                "total_images": int,
                "skipped_images": int,
                "imported_images": int,
                "label_counts": dict[str, int],
                "elapsed_sec": float,
                "rate_pps": float,  // pixels per second
                "manifest_updated": bool,  // 每张影像后已同步更新 manifest
            }
        """
        if not self.manager.health_check():
            raise ConnectionError(f"Qdrant 不可达: {self.manager.url}")

        if not self.manager.collection_exists():
            raise RuntimeError(
                f"Collection '{self.manager.collection_name}' 不存在，请先创建 Collection"
            )

        # 迁移 image_id 索引为 keyword：先检查当前 schema，已是 keyword 则跳过，
        # 仅当 text 或缺失时删除重建。保证 check_image_count 的 MatchValue 查询
        # 走索引路径（避免每次导入触发全量 payload 扫描），且对既有 collection
        # 自动生效（CLI 与 WebUI 共用本入口）。
        # 迁移是性能优化，失败不应阻塞导入：包 try/except 降级为告警（仅慢不中断）。
        try:
            self.manager.migrate_image_id_index()
        except Exception:  # noqa: BLE001 — 迁移失败降级为告警，不阻塞导入
            warnings.warn("image_id 索引迁移失败，check_image_count 可能仍走全量扫描")

        start_time = time.perf_counter()

        pairs = PixelDataLoader.scan_directory(Path(data_dir))
        if not pairs:
            print("警告: 未找到匹配的 SE/DW 文件对")
            return {
                "total_pixels": 0, "total_images": 0,
                "skipped_images": 0, "imported_images": 0,
                "label_counts": {}, "elapsed_sec": 0, "rate_pps": 0,
            }

        total_pixels = 0
        total_images = 0
        skipped_images = 0
        label_counts: dict[str, int] = {}

        # 预计算断点续传状态与待导入像素总数（仅当需要进度回调时）。
        # existing_counts 缓存 count 查询结果，主循环直接复用；
        # total 只累加真实待导入影像（16384/张），全部跳过时为 0。
        if progress_callback is not None:
            if no_resume:
                existing_counts = {pair.image_id: 0 for pair in pairs}
            else:
                existing_counts = {
                    pair.image_id: _retry_call(
                        PixelDataLoader.check_image_count, pair.image_id, self.manager
                    )
                    for pair in pairs
                }
            total = sum(
                128 * 128 for pair in pairs
                if no_resume or existing_counts[pair.image_id] < 128 * 128
            )
        else:
            existing_counts = {}
            total = 0

        # 进度推进器：由 import_directory 统一驱动，_batch_upsert 每批后回调
        imported_so_far = 0

        def _batch_progress(batch_len: int) -> None:
            nonlocal imported_so_far
            imported_so_far += batch_len
            progress_callback(imported_so_far, total)

        def _get_existing_count(pair: ImagePair) -> int:
            if no_resume:
                return 0
            if progress_callback is not None:
                return existing_counts[pair.image_id]
            return _retry_call(PixelDataLoader.check_image_count, pair.image_id, self.manager)

        # 使用 tqdm 进度条
        try:
            from tqdm import tqdm
            pair_iter = tqdm(pairs, desc="导入影像", unit="img")
        except ImportError:
            pair_iter = pairs

        for pair in pair_iter:
            total_images += 1
            pair_label_counts: dict[str, int] = {}

            # 断点续传：按 count 判定，而非二进制 set
            existing_count = _get_existing_count(pair)
            if not no_resume and existing_count >= 128 * 128:
                skipped_images += 1
                dw_data = PixelDataLoader.load_dw(pair.dw_path)
                unique, counts = np.unique(dw_data, return_counts=True)
                for lbl, cnt in zip(unique, counts):
                    name = LABEL_NAMES.get(int(lbl), "unknown")
                    pair_label_counts[name] = int(cnt)
                # 跳过：manifest 记录 count 值（>= 16384），避免跳过后清单被置空
                manifest_pixels = existing_count
            else:
                imported, skip = self.import_image_pair(
                    pair,
                    _batch_callback=_batch_progress if progress_callback is not None else None,
                )
                total_pixels += imported

                dw_data = PixelDataLoader.load_dw(pair.dw_path)
                unique, counts = np.unique(dw_data, return_counts=True)
                for lbl, cnt in zip(unique, counts):
                    name = LABEL_NAMES.get(int(lbl), "unknown")
                    pair_label_counts[name] = int(cnt)
                # 导入成功（含覆盖重传）记录 16384；import_image_pair 内部跳过
                # （如 no_resume 但 DB 已完整导入）时记录其返回的 count 值（skip）
                manifest_pixels = imported if imported > 0 else skip

            # 合并 label 统计
            for name, cnt in pair_label_counts.items():
                label_counts[name] = label_counts.get(name, 0) + cnt

            # 每张影像（导入或跳过）处理完成后同步更新 manifest（Design Doc §4.4 /
            # import-manifest spec）。无条件调用（不依赖 progress_callback），
            # 保证 CLI 与 WebUI 导入（共用本入口）都更新 manifest。
            update_manifest(pair.image_id, manifest_pixels, self.manager.collection_name)

        elapsed = time.perf_counter() - start_time

        # 索引重建：indexing_threshold=0 强制触发全量 HNSW 向量索引重建，
        # 使新导入向量快速进入可检索状态。
        # 仅重建 HNSW 向量索引，不重建 payload 标量索引（用户确认，避免重建窗口内过滤查询不可用）。
        if reindex:
            self.manager.client.update_collection(
                collection_name=self.manager.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )

        stats = {
            "total_pixels": total_pixels,
            "total_images": total_images,
            "skipped_images": skipped_images,
            "imported_images": total_images - skipped_images,
            "label_counts": label_counts,
            "elapsed_sec": elapsed,
            "rate_pps": total_pixels / elapsed if elapsed > 0 else 0,
            "manifest_updated": True,
        }
        self._stats = stats
        return stats
