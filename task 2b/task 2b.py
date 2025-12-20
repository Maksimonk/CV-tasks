from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging


@dataclass
class BBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


@dataclass
class Config:
    input_path: str
    output_dir: str
    screenshots_dir: str
    bbox: BBox
    height_threshold: float
    distance_center: Tuple[float, float, float]
    distance_radius: float
    class_filter: int
    viz: bool
    downsample_for_viz: int
    random_seed: int


def load_point_cloud(path: str) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 3:
        classes = np.zeros((data.shape[0], 1), dtype=int)
        data = np.concatenate([data, classes], axis=1)
    elif data.shape[1] >= 4:
        data = data[:, :4]
        data[:, 3] = data[:, 3].astype(int)
    else:
        raise ValueError("Input file must have at least 3 columns: x y z")

    return data.astype(float)


def save_point_cloud(path: str, points: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pts = np.column_stack((points[:, :3], points[:, 3].astype(int)))
    fmt = "%.6f %.6f %.6f %d"
    np.savetxt(path, pts, fmt=fmt)


def filter_by_bbox(points: np.ndarray, bbox: BBox) -> np.ndarray:
    mask = (
        (points[:, 0] >= bbox.xmin)
        & (points[:, 0] <= bbox.xmax)
        & (points[:, 1] >= bbox.ymin)
        & (points[:, 1] <= bbox.ymax)
        & (points[:, 2] >= bbox.zmin)
        & (points[:, 2] <= bbox.zmax)
    )
    return points[mask]


def filter_by_height(points: np.ndarray, z_threshold: float) -> np.ndarray:
    return points[points[:, 2] > z_threshold]


def filter_by_distance(
    points: np.ndarray, center: Tuple[float, float, float], radius: float
) -> np.ndarray:
    center = np.array(center)
    distances = np.linalg.norm(points[:, :3] - center.reshape(1, 3), axis=1)
    return points[distances <= radius]


def filter_by_class(points: np.ndarray, class_id: int) -> np.ndarray:
    return points[points[:, 3] == class_id]


def downsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    n = points.shape[0]
    if n <= max_points or max_points <= 0:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def show_cloud(
    points: np.ndarray,
    title: str = "Point Cloud",
    save_to: str = None,
    max_points: int = 10000,
    seed: int = 0,
):
    pts = downsample(points, max_points, seed)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    classes = pts[:, 3]
    unique = np.unique(classes)
    if unique.size > 1:
        im = ax.scatter(pts[:, 1], pts[:, 0], pts[:, 2], s=1, c=classes)
        fig.colorbar(im, ax=ax, label="class")
    else:
        ax.scatter(pts[:, 1], pts[:, 0], pts[:, 2], s=0.5)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=300)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("segment_pc")

    cfg = Config(
        input_path="terra_02_000004.asc",
        output_dir="./output",
        screenshots_dir="./скриншоты",
        bbox=BBox(
            xmin=-10.0,
            xmax=10.0,
            ymin=-10.0,
            ymax=10.0,
            zmin=0.0,
            zmax=20.0
        ),
        height_threshold=2.0,
        distance_center=(0.0, 0.0, 0.0),
        distance_radius=5.0,
        class_filter=-1,
        viz=True,
        downsample_for_viz=5000,
        random_seed=42
    )

    log.info("Configuration loaded")

    in_path = Path(cfg.input_path)
    out_dir = Path(cfg.output_dir)
    screenshots_dir = Path(cfg.screenshots_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        log.error("Input file does not exist: %s", in_path)
        return

    points = load_point_cloud(str(in_path))
    log.info("Loaded point cloud: %d points", points.shape[0])

    bbox_filtered = filter_by_bbox(points, cfg.bbox)
    log.info("BBox filtered: %d points", bbox_filtered.shape[0])
    save_point_cloud(out_dir / "bbox_filtered.asc", bbox_filtered)

    high_points = filter_by_height(points, cfg.height_threshold)
    log.info("High points (Z > %s): %d points", cfg.height_threshold, high_points.shape[0])
    save_point_cloud(out_dir / "high_points.asc", high_points)

    near_center = filter_by_distance(points, tuple(cfg.distance_center), cfg.distance_radius)
    log.info("Near center (radius=%s): %d points", cfg.distance_radius, near_center.shape[0])
    save_point_cloud(out_dir / "near_center.asc", near_center)

    if cfg.class_filter >= 0:
        class_pts = filter_by_class(points, cfg.class_filter)
        log.info("Class %d filtered: %d points", cfg.class_filter, class_pts.shape[0])
        save_point_cloud(out_dir / f"class_{cfg.class_filter}.asc", class_pts)

    if cfg.viz:
        log.info("Generating visualizations in folder: %s", screenshots_dir)
        show_cloud(
            points,
            title="Исходное облако точек",
            save_to=str(screenshots_dir / "Оригинальное облако точек.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )
        show_cloud(
            bbox_filtered,
            title="После фильтрации BBox",
            save_to=str(screenshots_dir / "Bounding box.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )
        show_cloud(
            high_points,
            title="Высокие точки (Z > %.1f)" % cfg.height_threshold,
            save_to=str(screenshots_dir / "Фильтр по высоте.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )
        show_cloud(
            near_center,
            title="Близкие к центру (радиус=%.1f)" % cfg.distance_radius,
            save_to=str(screenshots_dir / "Фильтр по расстоянию от точки.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )

    report_lines = [
        f"Исходное кол-во точек: {points.shape[0]}",
        f"После фильтрации BBox: {bbox_filtered.shape[0]}",
        f"Высокие точки (Z > {cfg.height_threshold}): {high_points.shape[0]}",
        f"Близкие к центру (радиус={cfg.distance_radius}): {near_center.shape[0]}",
        f"Скриншоты сохранены в: {screenshots_dir}",
    ]
    (out_dir / "report.txt").write_text("\n".join(report_lines))
    log.info("Отчёт сохранён в %s", out_dir / "report.txt")
    log.info("Скриншоты сохранены в %s", screenshots_dir)


if __name__ == "__main__":
    main()