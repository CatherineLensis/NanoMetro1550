"""
Модуль извлечения 9 инвариантных признаков формы, размеров и отражения 3D-кластера (Tier 2).
Признаки согласованы с One-Class моделью Махаланобиса (обученной на реальных данных тоннеля).
Все вычисления выполняются в центрированной физической системе координат кластера.
"""
from typing import Dict, Any, Tuple
import numpy as np

FEATURE_NAMES = [
    "linearity",        # Вытянутость формы (собственные числа PCA: (l1 - l2) / l1)
    "planarity",        # Планарность / плоскостность: (l2 - l3) / l1
    "sphericity",       # Объемность / сферичность: l3 / l1
    "extent_x",         # Поперечный габарит (м)
    "extent_y",         # Продольный габарит вдоль пути (м)
    "extent_z",         # Вертикальный габарит по высоте (м)
    "intensity_mean",   # Средняя интенсивность отражения лидара
    "intensity_std",    # Дисперсия / шероховатость интенсивности
    "mean_nn_dist",     # Средняя плотность точек (дистанция до ближайшего соседа)
]

TARGET_SAMPLE_POINTS = 256


def extract_cluster_features(
    cluster: Dict[str, Any],
    target_points: int = TARGET_SAMPLE_POINTS,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Вычисляет 9 признаков для словаря кластера, переданного из Tier 1.
    Гарантирует строгое соответствие схеме обучения:
    1. Центрированные физические координаты (x - x_mean, y - y_mean, z - z_mean).
    2. Нормализация размера выборки до target_points (256 точек) для сохранения масштаба плотности.
    3. Вычисление тензора инерции через PCA.
    """
    if "pts_centered_int" in cluster and cluster["pts_centered_int"].shape[1] >= 4:
        pts4 = cluster["pts_centered_int"]
        xyz = pts4[:, :3]
        intens = pts4[:, 3]
    elif "pts_xyz" in cluster:
        xyz = cluster["pts_xyz"]
        # Центрирование
        xyz = xyz - xyz.mean(axis=0)
        intens = cluster.get("pts_centered_int", np.zeros((len(xyz), 4)))[:, 3] if "pts_centered_int" in cluster else np.zeros(len(xyz), dtype=np.float32)
    elif "pts_dx_dy_dz_int" in cluster:
        pts4 = cluster["pts_dx_dy_dz_int"]
        xyz = pts4[:, :3] - pts4[:, :3].mean(axis=0)
        intens = pts4[:, 3]
    else:
        raise ValueError("Cluster dict does not contain valid point arrays")

    m = len(xyz)
    if m == 0:
        zero_vec = np.zeros(9, dtype=np.float32)
        zero_dict = {name: 0.0 for name in FEATURE_NAMES}
        return zero_vec, zero_dict

    # Сэмплирование до фиксированного числа точек (256) как при обучении One-Class
    rng = np.random.default_rng(seed)
    if m >= target_points:
        sample_idx = rng.choice(m, target_points, replace=False)
    else:
        sample_idx = rng.choice(m, target_points, replace=True)

    p_xyz = xyz[sample_idx]
    p_int = intens[sample_idx]

    # 1. Форма через ковариационную матрицу (PCA)
    cov = np.cov(p_xyz.T)
    if cov.ndim < 2:
        lin, plane, sph = 1.0, 0.0, 0.0
    else:
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        l1, l2, l3 = np.maximum(eigvals, 1e-9)
        lin = float((l1 - l2) / l1)
        plane = float((l2 - l3) / l1)
        sph = float(l3 / l1)

    # 2. Пространственные габариты полного кластера
    if "extent" in cluster:
        ext_x = float(cluster["extent"][0])
        ext_y = float(cluster["extent"][1])
        ext_z = float(cluster["extent"][2])
    else:
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        ext_x, ext_y, ext_z = float(extent[0]), float(extent[1]), float(extent[2])

    # 3. Статистика интенсивности
    i_mean = float(p_int.mean()) if len(p_int) else 0.0
    i_std = float(p_int.std()) if len(p_int) else 0.0

    # 4. Средняя дистанция ближайшего соседа (масштаб плотности согласован с выборкой 256 точек)
    dists = np.linalg.norm(p_xyz[:, None, :] - p_xyz[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    mean_nn = float(dists.min(axis=1).mean())

    feat_vector = np.array([lin, plane, sph, ext_x, ext_y, ext_z, i_mean, i_std, mean_nn], dtype=np.float32)
    feat_dict = {
        "linearity": lin,
        "planarity": plane,
        "sphericity": sph,
        "extent_x": ext_x,
        "extent_y": ext_y,
        "extent_z": ext_z,
        "intensity_mean": i_mean,
        "intensity_std": i_std,
        "mean_nn_dist": mean_nn,
    }
    return feat_vector, feat_dict
