"""
Классификатор и скорер аномалий Tier 2 (Project 14).
Реализует гибридный подход:
1. Семантические правила инфраструктуры метрополитена на базе ручной разметки (контррельс, кабель, платформа, шкафы СЦБ).
2. One-Class модель расстояния Махаланобиса, обученная исключительно на реальном фоне тоннеля (без утечки синтетики).
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np

from hybrid_clearance_solution.tier2_feature_extractor import extract_cluster_features, FEATURE_NAMES


class Tier2Classifier:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            curr_dir = os.path.dirname(__file__)
            model_path = os.path.join(curr_dir, "models", "tier2_oneclass_model_full.npz")

        if os.path.exists(model_path):
            data = np.load(model_path, allow_pickle=True)
            self.mu = data["mu"]
            self.cov_inv = data["cov_inv"]
            self.threshold = float(data["threshold"][0])
            self.feature_names = list(data["feature_names"])
            self.is_loaded = True
        else:
            self.mu = None
            self.cov_inv = None
            self.threshold = 4.655
            self.feature_names = FEATURE_NAMES
            self.is_loaded = False

    def classify_cluster(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Классифицирует кластер кандидата в препятствия.
        Возвращает:
        - is_obstacle: True (опасное препятствие) / False (штатная инфраструктура пути)
        - classification: 'hazard', 'person', 'guard_rail_limiter', 'cable', 'station_platform', 'electrical_cabinet', 'ballast_floor', 'benign_background'
        - anomaly_score: расстояние Махаланобиса
        - confidence: уверенность от 0.0 до 1.0
        - features: вычисленные 9 признаков
        """
        feat_vec, feat_dict = extract_cluster_features(cluster)

        lin = feat_dict["linearity"]
        plane = feat_dict["planarity"]
        sph = feat_dict["sphericity"]
        ext_x = feat_dict["extent_x"]
        ext_y = feat_dict["extent_y"]
        ext_z = feat_dict["extent_z"]
        mean_dz = cluster.get("mean_dz", 0.0)
        mean_dx = cluster.get("mean_dx", 0.0)

        # 1. Проверка семантических сигнатур инфраструктуры пути (разметка KatLab)

        # 1.1. Контррельс / ограничитель пути (guard_rail_limiter, класс 10)
        if (0.10 <= mean_dz <= 0.65) and (mean_dx >= 0.75) and (lin >= 0.60 or ext_y >= 1.2) and (ext_z <= 0.65):
            return {
                "is_obstacle": False,
                "classification": "guard_rail_limiter",
                "anomaly_score": 1.2,
                "confidence": 0.96,
                "features": feat_dict,
            }

        # 1.2. Продольный кабель / лоток / труба вдоль пути (cable, класс 9)
        if (lin >= 0.70) and (ext_y >= 1.5) and (ext_x <= 0.50):
            return {
                "is_obstacle": False,
                "classification": "cable",
                "anomaly_score": 1.5,
                "confidence": 0.94,
                "features": feat_dict,
            }

        # 1.3. Путевой шкаф СЦБ / релейный ящик / настенная аппаратура (electrical_cabinet / wall_fixture, класс 7)
        # Находится у кромки габарита (|dx| >= 1.05м), вытянут вдоль стены (ext_z до 3.2м), малая толщина (ext_x <= 0.60м), плоский/линейный (sph <= 0.05)
        if (mean_dx >= 1.05) and (0.50 <= ext_z <= 3.30) and (ext_x <= 0.60) and (sph <= 0.05) and (lin >= 0.60 or plane >= 0.40):
            return {
                "is_obstacle": False,
                "classification": "electrical_cabinet",
                "anomaly_score": 2.2,
                "confidence": 0.93,
                "features": feat_dict,
            }

        # 1.4. Край платформы станции / кожух контактного рельса (station_platform_edge / contact_rail_cover, класс 5 / 12)
        # Попытка поднять mean_dx с 0.75 до 1.35 (чтобы устранить мерцание hazard/station_platform_edge
        # на приближающемся препятствии validation-бага, кадры 463-479) откатана: честная проверка на
        # целевой зоне основного датасета (чанки 103-134) показала рост 60/510 (11.8%) -> 79/510 (15.5%)
        # -- то же самое реальное межпутевое препятствие, которое эта зона проверяет, стало течь через
        # ослабленное правило. FPR на реальных данных важнее устранения мерцания на синтетике.
        if (mean_dx >= 0.75) and (ext_x <= 0.50) and (sph <= 0.06) and (lin >= 0.50 or plane >= 0.35) and (ext_z <= 0.80):
            return {
                "is_obstacle": False,
                "classification": "station_platform_edge",
                "anomaly_score": 1.9,
                "confidence": 0.92,
                "features": feat_dict,
            }

        # 1.5. Балластная призма / шпалы (ballast_floor, класс 3)
        if (mean_dz <= 0.28) and (ext_z <= 0.25) and (plane >= 0.35 or lin >= 0.60):
            return {
                "is_obstacle": False,
                "classification": "ballast_floor",
                "anomaly_score": 1.8,
                "confidence": 0.95,
                "features": feat_dict,
            }

        # 1.6. Межпутевая стена/опора (pillar_support, класс 4) -- широкая структура,
        # которую высота не отличает от препятствия (высотный потолок NEAR_ZONE_MAX_OBSTACLE_HEIGHT
        # в tier1_curvilinear_detector.py оказался недостаточен: честная проверка на chunk 104,
        # 128-133 показала, что основная масса точек лежит НИЖЕ этого потолка, 0.22-2.3м --
        # ровно в диапазоне роста человека). Отличается по ШИРИНЕ: собственное правило is_person
        # этого файла ограничивает компактный объект ext_x<=1.20м; здесь ext_x доходит до ~1.4м
        # (почти вся полуширина габарита), что физически не похоже ни на человека, ни на груз/
        # оборудование -- похоже на протяжённую вдоль пути стену/опору.
        if (ext_x >= 1.30 or ext_y >= 1.80) and not (0.25 <= ext_x <= 1.20 and 0.25 <= ext_y <= 1.20):
            return {
                "is_obstacle": False,
                "classification": "pillar_support",
                "anomaly_score": 1.6,
                "confidence": 0.88,
                "features": feat_dict,
            }

        # 2. Оценка расстояния Махаланобиса (One-Class модель, обученная на реальных негативах)
        anomaly_score = 0.0
        if self.is_loaded and self.mu is not None and self.cov_inv is not None:
            diff = feat_vec - self.mu
            anomaly_score = float(np.sqrt(diff @ self.cov_inv @ diff.T))
        else:
            anomaly_score = 5.0

        is_hazard = (anomaly_score > self.threshold)

        # 3. Детекция человека (объемный вертикальный объект роста 0.8-2.2м на пути)
        is_person = (
            (0.25 <= ext_x <= 1.20)
            and (0.25 <= ext_y <= 1.20)
            and (0.75 <= ext_z <= 2.20)
            and (sph >= 0.04 or (lin < 0.85 and plane > 0.15))
            and (mean_dx <= 1.35)
        )

        if is_person:
            is_hazard = True
            tag = "person"
            conf = 0.98
        elif is_hazard:
            tag = "hazard"
            conf = min(0.99, float(0.5 + 0.5 * (anomaly_score / (self.threshold * 1.5))))
        else:
            tag = "benign_background"
            conf = min(0.95, float(1.0 - 0.5 * (anomaly_score / self.threshold)))

        return {
            "is_obstacle": is_hazard,
            "classification": tag,
            "anomaly_score": anomaly_score,
            "confidence": conf,
            "features": feat_dict,
        }

    def filter_clusters(self, clusters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool, Optional[float]]:
        """
        Обрабатывает список кластеров от Tier 1.
        Возвращает:
        - отфильтрованные подтвержденные препятствия
        - флаг подтвержденной опасности
        - минимальная дистанция до подтвержденного препятствия
        """
        confirmed_obstacles = []
        min_dist = None

        for c in clusters:
            res = self.classify_cluster(c)
            c["tier2"] = res
            if res["is_obstacle"]:
                confirmed_obstacles.append(c)
                dist = c.get("min_distance", None)
                if dist is not None:
                    if min_dist is None or dist < min_dist:
                        min_dist = dist

        has_obstacle = len(confirmed_obstacles) > 0
        return confirmed_obstacles, has_obstacle, min_dist
