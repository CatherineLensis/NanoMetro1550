"""
Единый гибридный конвейер обнаружения препятствий (Project 14).
Объединяет:
- Префильтрацию артефактов и валидацию облака
- Tier 1: Криволинейный адаптивный габарит ГОСТ 9238 (2.5-45м) + Боровой монитор (45-200м)
- Tier 2: Инвариантный анализ геометрических дескрипторов + One-Class Mahalanobis + Фильтр инфраструктуры
- Формирование 3D Bounding Box маркеров для визуализации в RViz2
"""
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from hybrid_clearance_solution.tier1_curvilinear_detector import Tier1CurvilinearDetector
from hybrid_clearance_solution.tier2_classifier import Tier2Classifier


class HybridPerceptionPipeline:
    def __init__(
        self,
        model_path: Optional[str] = None,
        slice_step: float = 1.0,
        y_max: float = -2.5,
        y_min: float = -60.0,
        lateral_margin: float = 1.45,
        warmup_frames: int = 8,
    ):
        self.tier1 = Tier1CurvilinearDetector(
            slice_step=slice_step,
            y_max=y_max,
            y_min=y_min,
            body_half_width=lateral_margin,
            warmup_frames=warmup_frames,
        )
        self.tier2 = Tier2Classifier(model_path=model_path)
        self.last_timestamp: Optional[float] = None
        self.frame_index = 0
        self.slice_step = slice_step
        self.y_max = y_max
        self.y_min = y_min
        self.lateral_margin = lateral_margin
        self.warmup_frames = warmup_frames

    def reset(self):
        """Сброс внутреннего состояния при смене bag-файла или временном разрыве."""
        self.tier1 = Tier1CurvilinearDetector(
            slice_step=self.slice_step,
            y_max=self.y_max,
            y_min=self.y_min,
            body_half_width=self.lateral_margin,
            warmup_frames=self.warmup_frames,
        )
        self.last_timestamp = None
        self.frame_index = 0

    def process_point_cloud(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        ring: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Основной метод обработки одного кадра PointCloud2.
        Гарантирует выполнение < 35 мс на стандартном CPU.
        """
        t_start = time.perf_counter()

        # Проверка временного разрыва (если между кадрами прошло более 1.0 с)
        if timestamp is not None:
            if self.last_timestamp is not None and (timestamp - self.last_timestamp > 1.0):
                self.reset()
            self.last_timestamp = timestamp

        # --- ЭТАП 1: Tier 1 Геометрия и Боровой монитор ---
        r1 = self.tier1.process_frame(x, y, z, intensity=intensity, ring=ring)

        # --- ЭТАП 2: Tier 2 Легковесная классификация кандидатов ---
        raw_clusters = r1.get("clusters", [])
        confirmed_obstacles, has_envelope_hazard, min_envelope_dist = self.tier2.filter_clusters(raw_clusters)

        # Проверка блокировки борового монитора дальней зоны
        bore_clear = r1["bore_clear"]
        bore_range = r1["bore_range"]

        # Итоговое решение
        obstacle_detected = False
        hazard_dist = None
        hazard_type = "NONE"
        threat_level = "NONE"

        if has_envelope_hazard:
            obstacle_detected = True
            hazard_dist = min_envelope_dist
            # Определим тип по первому опасному кластеру
            primary_cls = confirmed_obstacles[0]["tier2"]["classification"]
            hazard_type = "PERSON" if primary_cls == "person" else "CLEARANCE_INTRUSION"
            threat_level = "CRITICAL" if (hazard_dist is not None and hazard_dist < 35.0) else "WARNING"
        elif not bore_clear:
            obstacle_detected = True
            hazard_dist = bore_range
            hazard_type = "BORE_OCCLUSION_LONG_RANGE"
            threat_level = "WARNING"

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.frame_index += 1

        # Формирование 3D bounding boxes для публикации в RViz
        boxes = []
        for c in confirmed_obstacles:
            cent = c["centroid"]
            ext = c["extent"]
            boxes.append({
                "x": cent[0], "y": cent[1], "z": cent[2],
                "size_x": max(0.3, ext[0]),
                "size_y": max(0.4, ext[1]),
                "size_z": max(0.4, ext[2]),
                "class": c["tier2"]["classification"],
                "score": c["tier2"]["anomaly_score"],
            })

        return {
            "obstacle_detected": obstacle_detected,
            "obstacle_distance": float(hazard_dist) if hazard_dist is not None else None,
            "threat_level": threat_level,
            "hazard_type": hazard_type,
            "n_confirmed_hazards": len(confirmed_obstacles),
            "confirmed_boxes": boxes,
            "bore_clear": bore_clear,
            "bore_range": bore_range,
            "latency_ms": latency_ms,
            "fps": 1000.0 / max(0.01, latency_ms),
            "tier1_candidate_points": r1["n_candidate_points"],
            "frame_index": self.frame_index,
        }
