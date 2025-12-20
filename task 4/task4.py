from dataclasses import dataclass
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


@dataclass
class Config:
    N: int = 1000
    seed: int = 42
    save_dir: str = "./outputs"
    plot_matplotlib: bool = True
    plot_plotly: bool = True
    matplotlib_marker_size: int = 10
    plotly_marker_size: int = 3
    save_csv: bool = True
    save_ply: bool = False
    use_alt_colormap: bool = True
    alt_colormap: str = "viridis"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pointcloud_csv(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    header = "x,y,z,r,g,b"
    data = np.hstack((points, colors))
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_pointcloud_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    n = points.shape[0]
    colors255 = np.clip((colors * 255).astype(int), 0, 255)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors255):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def main() -> None:
    # Конфигурация (можно менять параметры здесь)
    cfg = Config(
        N=1000,
        seed=42,
        save_dir="./outputs",
        plot_matplotlib=True,
        plot_plotly=False,  # plotly требует установки plotly библиотеки
        matplotlib_marker_size=10,
        plotly_marker_size=3,
        save_csv=True,
        save_ply=False,
        use_alt_colormap=True,
        alt_colormap="viridis"
    )
    
    print("Конфигурация:")
    print(f"N (количество точек): {cfg.N}")
    print(f"Seed: {cfg.seed}")
    print(f"Папка сохранения: {cfg.save_dir}")
    print(f"Визуализация matplotlib: {cfg.plot_matplotlib}")
    print(f"Визуализация plotly: {cfg.plot_plotly}")

    np.random.seed(cfg.seed)
    N = cfg.N
    points = np.random.rand(N, 3)
    colors = points.copy()

    alt_colors = None
    if cfg.use_alt_colormap:
        center = np.array([0.5, 0.5, 0.5])
        dist = np.linalg.norm(points - center, axis=1)
        norm_dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-12)
        cmap = cm.get_cmap(cfg.alt_colormap)
        alt_colors = cmap(norm_dist)[:, :3]

    ensure_dir(cfg.save_dir)
    print(f"\nРезультаты будут сохранены в: {cfg.save_dir}")

    if cfg.save_csv:
        csv_path = os.path.join(cfg.save_dir, "points_rgb.csv")
        save_pointcloud_csv(csv_path, points, colors)
        print(f"✓ Сохранено CSV: {csv_path}")

    if cfg.save_ply:
        ply_path = os.path.join(cfg.save_dir, "points_rgb.ply")
        save_pointcloud_ply(ply_path, points, colors)
        print(f"✓ Сохранено PLY: {ply_path}")

    if cfg.plot_matplotlib:
        # 2D визуализация с основными цветами
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(points[:, 0], points[:, 1], c=colors, s=cfg.matplotlib_marker_size)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D визуализация облака точек (RGB цвета)")
        ax.grid(True)

        out_png = os.path.join(cfg.save_dir, "points_2d_matplotlib.png")
        out_svg = os.path.join(cfg.save_dir, "points_2d_matplotlib.svg")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        fig.savefig(out_svg)
        plt.close(fig)
        print(f"✓ Сохранено Matplotlib (PNG): {out_png}")
        print(f"✓ Сохранено Matplotlib (SVG): {out_svg}")

        # 3D визуализация
        fig3d = plt.figure(figsize=(8, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        scatter3d = ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                c=colors, s=cfg.matplotlib_marker_size)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title("3D визуализация облака точек (Matplotlib)")
        
        out_png_3d = os.path.join(cfg.save_dir, "points_3d_matplotlib.png")
        fig3d.savefig(out_png_3d, dpi=200)
        plt.close(fig3d)
        print(f"✓ Сохранено 3D Matplotlib (PNG): {out_png_3d}")

        if alt_colors is not None:
            # 2D визуализация с альтернативной цветовой схемой
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(points[:, 0], points[:, 1], c=alt_colors, s=cfg.matplotlib_marker_size)
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_title(f"2D визуализация — {cfg.alt_colormap} (по расстоянию от центра)")
            ax2.grid(True)
            
            out_png2 = os.path.join(cfg.save_dir, "points_2d_alt_matplotlib.png")
            fig2.tight_layout()
            fig2.savefig(out_png2, dpi=200)
            plt.close(fig2)
            print(f"✓ Сохранено Matplotlib (альт, PNG): {out_png2}")

            # 3D визуализация с альтернативной цветовой схемой
            fig3d_alt = plt.figure(figsize=(8, 6))
            ax3d_alt = fig3d_alt.add_subplot(111, projection='3d')
            ax3d_alt.scatter(points[:, 0], points[:, 1], points[:, 2], 
                            c=alt_colors, s=cfg.matplotlib_marker_size)
            ax3d_alt.set_xlabel("X")
            ax3d_alt.set_ylabel("Y")
            ax3d_alt.set_zlabel("Z")
            ax3d_alt.set_title(f"3D визуализация — {cfg.alt_colormap}")
            
            out_png_3d_alt = os.path.join(cfg.save_dir, "points_3d_alt_matplotlib.png")
            fig3d_alt.savefig(out_png_3d_alt, dpi=200)
            plt.close(fig3d_alt)
            print(f"✓ Сохранено 3D Matplotlib (альт, PNG): {out_png_3d_alt}")

    if cfg.plot_plotly:
        try:
            import plotly.graph_objects as go
            print("\nЗагрузка Plotly...")
            
            rgb_strings = [
                f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors
            ]
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode="markers",
                        marker=dict(size=cfg.plotly_marker_size, color=rgb_strings),
                    )
                ]
            )
            fig.update_layout(
                title="3D визуализация облака точек (Plotly)",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            )

            out_html = os.path.join(cfg.save_dir, "points_3d_plotly.html")
            fig.write_html(out_html, include_plotlyjs="cdn")
            print(f"✓ Сохранено Plotly (HTML): {out_html}")

            if alt_colors is not None:
                rgb_strings_alt = [
                    f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                    for r, g, b in alt_colors
                ]
                fig_alt = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=points[:, 0],
                            y=points[:, 1],
                            z=points[:, 2],
                            mode="markers",
                            marker=dict(size=cfg.plotly_marker_size, color=rgb_strings_alt),
                        )
                    ]
                )
                fig_alt.update_layout(
                    title=f"3D (альт) — {cfg.alt_colormap}",
                    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                )
                out_html_alt = os.path.join(cfg.save_dir, "points_3d_plotly_alt.html")
                fig_alt.write_html(out_html_alt, include_plotlyjs="cdn")
                print(f"✓ Сохранено Plotly (альт HTML): {out_html_alt}")
                
        except ImportError:
            print("\n⚠ Plotly не установлен. Для использования plotly выполните:")
            print("pip install plotly")
            print("Или установите plotly==5.18.0")

    print(f"\n Все файлы сохранены в папке: {cfg.save_dir}")
    
    # Показываем итоговую картинку
    if cfg.plot_matplotlib:
        fig_final, ax_final = plt.subplots(figsize=(6, 6))
        final_colors = alt_colors if alt_colors is not None else colors
        ax_final.scatter(points[:, 0], points[:, 1], c=final_colors, s=cfg.matplotlib_marker_size)
        ax_final.set_xlabel("X")
        ax_final.set_ylabel("Y")
        ax_final.set_title(f"Итоговая визуализация ({cfg.N} точек)")
        ax_final.grid(True)
        plt.show()


if __name__ == "__main__":
    main()