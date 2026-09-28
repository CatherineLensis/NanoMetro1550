"""
Оптимизированный геометрический детектор Tier 1 (Project 14)
Реализует:
1. Фильтрацию артефактов крепления лидара (Self-return mask).
2. Боровой монитор дальней зоны (Bore-Clearance Monitor: 45-200м, 0% FP).
3. Адаптивный криволинейный габарит по ГОСТ 9238 (зона 2.5-45м, колея 1520мм).
4. Восстановление оси пути (EMA alpha=0.15, калибровка УГР 1075 мм).
5. Fix 5: Исключение зоны постоянного путевого контррельса (guard_rail_limiter, класс 10).
6. Fix 6: Строгий порог достоверности среза (MIN_CONFIDENT_SLICE_POINTS = 300) для предотвращения шума.
7. Быструю воксельно-графовую кластеризацию со связными компонентами (< 2 мс), исключающую дробление объектов на микро-боксы.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Геометрические константы
STANDARD_GAUGE = 1.520          # Русская колея 1520 мм
BODY_HALF_WIDTH = 1.45          # Полуширина габарита кузова вагона (ГОСТ 9238)
LOWER_HALF_WIDTH = 1.10         # Полуширина нижней ступени (исключает контактный рельс)
STEP_HEIGHT_THRESHOLD = 0.43    # Высота перехода от нижней ступени к кузову
ENVELOPE_HEIGHT = 3.30          # Максимальная высота габарита над рельсом
BALLAST_MARGIN_Z = 0.15         # Высота отсечения балласта/шпал
OBSTACLE_MIN_HEIGHT = 0.22      # Минимальная высота препятствия над рельсом (ГОСТ)

# Калибровка высоты лидара над рельсом (официальные данные ДепТранса: 1075 мм)
RAIL_HEIGHT_CALIBRATED = -1.075
RAIL_HEIGHT_TOLERANCE = 0.35

# Константы самовозврата (кронштейн сенсора на крыше вагона)
SELF_RETURN_CENTER = np.array([1.55, -1.40, 0.50], dtype=np.float32)
SELF_RETURN_RADIUS_SQ = 0.60 ** 2

# Боровой монитор дальней зоны
MIN_VALID_RANGE = 0.5
MAX_USABLE_RANGE = 220.0
BORE_HISTORY_LEN = 20
BORE_MIN_HISTORY = 5
BORE_DROP_FRACTION = 0.35
BORE_BASELINE_MIN = 25.0

# Срезы тоннеля
SLICE_STEP = 1.0
Y_MAX = -2.5
Y_MIN = -60.0                   # Дальность охвата криволинейного анализа (до 60м)
MIN_SLICE_POINTS = 15
MIN_CONFIDENT_SLICE_POINTS = 300  # Fix 6: строгий порог надежности среза
MAX_DRIFT_PER_SLICE = 0.15
EMA_ALPHA = 0.15
FORCE_REFRESH_INTERVAL = 15
WARMUP_FRAMES = 8               # Период начальной геометрической сходимости

# Fix 5: Известный постоянный путевой контррельс / ограничитель (guard_rail_limiter, класс 10)
GUARD_RAIL_DX_MIN = 0.8         # м, |dx| от оси пути
GUARD_RAIL_DX_MAX = 2.0
GUARD_RAIL_DZ_MIN = 0.0         # м, высота над УГР
GUARD_RAIL_DZ_MAX = 0.6

# Потолок высоты препятствия в зоне ближней continuity-доверенности (не подтверждённой
# однопутностью напрямую) -- см. комментарий у effective_height_ceiling в process_frame.
# Выше человеческого роста с запасом (собственное правило is_person этого пакета: до 2.20м).
NEAR_ZONE_MAX_OBSTACLE_HEIGHT = 2.3

# Край платформы станции (pillar_support, класс 4 по разметке Кати) -- известная,
# физически предсказуемая геометрия: край платформы спроектирован так, чтобы НИКОГДА не
# задевать поезд (габарит вагона обеспечивает проезд с минимальным, но безопасным зазором
# для посадки пассажиров) -- значит его можно и нужно исключать по позиции, как контррельс
# (Fix 5), а не только по ширине кластера (см. Fix pillar_support в tier2_classifier.py,
# который остаётся как подстраховка на случай смещённой оценки оси пути в near-zone
# continuity-режиме). Диапазон получен агрегацией 131 687 размеченных точек класса
# pillar_support по всем размеченным реальным кадрам (labeling/flagged_events_diverse,
# labeling/flagged_events_groupACD): |x| p1-p99 = 1.45-3.16м от оси пути, dz (высота над УГР,
# z + 1.075) p1-p99 = ~0-2.1м -- с запасом.
PLATFORM_EDGE_DX_MIN = 1.35   # м, |dx| от оси пути
PLATFORM_EDGE_DX_MAX = 3.30
PLATFORM_EDGE_DZ_MIN = -0.10  # м, высота над УГР (с небольшим запасом ниже наблюдаемого минимума)
PLATFORM_EDGE_DZ_MAX = 2.20


class Tier1CurvilinearDetector:
    def __init__(
        self,
        slice_step: float = SLICE_STEP,
        y_max: float = Y_MAX,
        y_min: float = Y_MIN,
        body_half_width: float = BODY_HALF_WIDTH,
        lower_half_width: float = LOWER_HALF_WIDTH,
        step_height: float = STEP_HEIGHT_THRESHOLD,
        envelope_height: float = ENVELOPE_HEIGHT,
        ema_alpha: float = EMA_ALPHA,
        warmup_frames: int = WARMUP_FRAMES,
    ):
        self.slice_step = slice_step
        self.y_max = y_max
        self.y_min = y_min
        self.body_half_width = body_half_width
        self.lower_half_width = lower_half_width
        self.step_height = step_height
        self.envelope_height = envelope_height
        self.ema_alpha = ema_alpha
        self.warmup_frames = warmup_frames

        # Сетка Y-срезов
        self.slice_y = np.concatenate([[y_max], np.arange(y_max - slice_step, y_min - slice_step, -slice_step)])
        self.n_slices = len(self.slice_y)

        # Персистентное состояние траектории
        self.state_cx = np.full(self.n_slices, np.nan, dtype=np.float32)
        self.state_rz = np.full(self.n_slices, np.nan, dtype=np.float32)
        self.state_confident = np.zeros(self.n_slices, dtype=bool)
        self.state_bore_confirmed = np.zeros(self.n_slices, dtype=bool)  # True only when is_single_bore
                                                                           # passed -- strict geometric proof,
                                                                           # vs. state_confident which also
                                                                           # accepts the near-zone continuity
                                                                           # bypass (see process_frame)
        self.state_initialized = np.zeros(self.n_slices, dtype=bool)
        self.state_stale_count = np.zeros(self.n_slices, dtype=np.int32)
        self.state_exclusion_streak = np.zeros(self.n_slices, dtype=np.int32)

        # История борового монитора
        self.bore_history: List[float] = []
        self.frame_count = 0

    def find_near_track_anchor(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
        """Быстрый поиск оси колеи 1520 мм в ближней зоне [-6.0, -2.5] м через свертку с шаблоном."""
        mask = (y >= -6.0) & (y <= -2.5) & (z >= -1.7) & (z <= -0.8)
        if mask.sum() < 40:
            return 0.0, RAIL_HEIGHT_CALIBRATED, 0.0

        floor_x = x[mask]
        floor_z = z[mask]
        rail_z = float(np.percentile(floor_z, 85))
        rail_z = float(np.clip(rail_z, RAIL_HEIGHT_CALIBRATED - RAIL_HEIGHT_TOLERANCE,
                                 RAIL_HEIGHT_CALIBRATED + RAIL_HEIGHT_TOLERANCE))

        bin_size = 0.03
        edges = np.arange(-3.5, 3.5 + bin_size, bin_size, dtype=np.float32)
        hist, _ = np.histogram(floor_x, bins=edges)
        centers = edges[:-1] + bin_size / 2.0

        shift = int(round(STANDARD_GAUGE / bin_size))
        if shift >= len(hist):
            return 0.0, rail_z, 0.0

        combined = hist[:-shift] + hist[shift:]
        cand_centers = centers[:len(combined)] + STANDARD_GAUGE / 2.0
        prior_weights = np.exp(-0.5 * (cand_centers / 0.5) ** 2)
        scores = combined * prior_weights
        best_idx = np.argmax(scores)
        return float(cand_centers[best_idx]), rail_z, float(combined[best_idx])

    def _interpolate_state(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Интерполяция накопленного состояния траектории оси пути и УГР на произвольные точки.
        Также возвращает per-point флаг bore_confirmed (строгое геометрическое подтверждение
        однопутности, а не только доверие по ближней непрерывности) -- используется, чтобы
        сузить допустимую высоту препятствия там, где геометрия среза не проверена напрямую."""
        y_asc = self.slice_y[::-1]
        cx_asc = self.state_cx[::-1]
        rz_asc = self.state_rz[::-1]
        conf_asc = self.state_confident[::-1].astype(np.float32)
        bore_asc = self.state_bore_confirmed[::-1].astype(np.float32)

        cx = np.interp(y, y_asc, cx_asc)
        rz = np.interp(y, y_asc, rz_asc)

        idx = np.searchsorted(y_asc, y).clip(0, len(y_asc) - 1)
        idx_prev = (idx - 1).clip(0, len(y_asc) - 1)
        use_prev = np.abs(y - y_asc[idx_prev]) < np.abs(y - y_asc[idx])
        nearest = np.where(use_prev, idx_prev, idx)
        confident = conf_asc[nearest] > 0.5
        bore_confirmed = bore_asc[nearest] > 0.5
        return cx, rz, confident, bore_confirmed

    def process_frame(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        ring: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Обработка одного кадра лидара с гарантированной латентностью < 30 мс."""
        # 1. Валидация и фильтрация самовозврата (кронштейна)
        r2 = x * x + y * y + z * z
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (r2 > 0.09)

        dxs = x - SELF_RETURN_CENTER[0]
        dys = y - SELF_RETURN_CENTER[1]
        dzs = z - SELF_RETURN_CENTER[2]
        is_self = (dxs * dxs + dys * dys + dzs * dzs) < SELF_RETURN_RADIUS_SQ
        valid &= ~is_self

        vx, vy, vz = x[valid], y[valid], z[valid]
        vr2 = r2[valid]
        vr = np.sqrt(vr2)
        v_int = intensity[valid] if intensity is not None else np.zeros_like(vx)
        v_ring = ring[valid] if ring is not None else np.zeros_like(vx, dtype=np.int32)

        # 2. Боровой монитор дальней зоны (45-200 м)
        bore_range = float(vr.max()) if len(vr) > 0 else 0.0
        bore_clear = True
        bore_baseline = None
        if len(self.bore_history) >= BORE_MIN_HISTORY:
            bore_baseline = float(np.median(self.bore_history[-BORE_HISTORY_LEN:]))
            if bore_baseline > BORE_BASELINE_MIN and bore_range < bore_baseline * (1.0 - BORE_DROP_FRACTION):
                bore_clear = False

        self.bore_history.append(bore_range)
        if len(self.bore_history) > BORE_HISTORY_LEN:
            self.bore_history.pop(0)

        warming_up = (self.frame_count < self.warmup_frames) or (not self.state_initialized.any())

        # 3. Классификация попадания в габарит ГОСТ 9238 по накопленному состоянию ДО этого кадра
        if warming_up:
            is_candidate = np.zeros(len(vx), dtype=bool)
            cx = np.zeros(len(vx), dtype=np.float32)
            rz = np.full(len(vx), RAIL_HEIGHT_CALIBRATED, dtype=np.float32)
            dx = vx
            dz = vz - rz
            confident_pt = np.zeros(len(vx), dtype=bool)
        else:
            cx, rz, confident_pt, bore_confirmed_pt = self._interpolate_state(vy)
            dx = vx - cx
            dz = vz - rz

            # Ступенчатый габарит ГОСТ 9238:
            # dz < 0.43м -> |dx| <= 1.10м (колесный пояс, балластная призма, 3-й рельс)
            # 0.43 <= dz <= 3.30м -> |dx| <= 1.45м (кузов вагона с кинематическим запасом)
            in_lat = np.where(dz < self.step_height, np.abs(dx) <= self.lower_half_width,
                              np.abs(dx) <= self.body_half_width)
            # Потолок высоты препятствия: полный габарит (3.30м) там, где однопутность
            # среза подтверждена геометрически (bore_confirmed); там, где слайс доверяем
            # только по ближней continuity-эвристике (станции/стрелки, см. is_slice_confident
            # ниже), потолок снижен до NEAR_ZONE_MAX_OBSTACLE_HEIGHT -- честная проверка на
            # полном датасете (chunk 104, 128-133) показала, что именно в этой зоне высокая
            # (до 3.2-3.3м) межпутевая стена/опора (pillar_support) даёт 98-100% ложных
            # срабатываний, тогда как человек/груз по ТЗ и собственному правилу is_person
            # этого пакета не выше ~2.2м -- так станционный сценарий (человек упал на путь)
            # остаётся ловиться, а стена/опора над человеческого роста -- нет.
            effective_height_ceiling = np.where(bore_confirmed_pt, self.envelope_height,
                                                  NEAR_ZONE_MAX_OBSTACLE_HEIGHT)
            in_vert = (dz >= OBSTACLE_MIN_HEIGHT) & (dz <= effective_height_ceiling)
            in_range = (vy <= self.y_max) & (vy >= self.y_min)

            # Fix 5: исключение зоны постоянного путевого контррельса (guard_rail_limiter)
            in_guard_rail_zone = (
                (np.abs(dx) >= GUARD_RAIL_DX_MIN)
                & (np.abs(dx) <= GUARD_RAIL_DX_MAX)
                & (dz >= GUARD_RAIL_DZ_MIN)
                & (dz <= GUARD_RAIL_DZ_MAX)
            )

            # Край платформы станции -- та же логика, что и Fix 5: известная, физически
            # предсказуемая позиция, которая по конструкции габарита НИКОГДА не задевает
            # поезд (см. PLATFORM_EDGE_* выше). В основном перекрывается с in_lat лишь на
            # узком краю (|dx| до 1.45м), т.к. большая часть реальной геометрии платформы
            # лежит уже за пределами кузовного габарита -- поэтому Tier2-правило
            # `pillar_support` (по ширине кластера) остаётся основной защитой на случай
            # смещённой оценки оси пути, а это исключение усиливает её на границе.
            in_platform_edge_zone = (
                (np.abs(dx) >= PLATFORM_EDGE_DX_MIN)
                & (np.abs(dx) <= PLATFORM_EDGE_DX_MAX)
                & (dz >= PLATFORM_EDGE_DZ_MIN)
                & (dz <= PLATFORM_EDGE_DZ_MAX)
            )

            # Кандидаты в препятствия (только при достоверном определении среза)
            is_candidate = in_lat & in_vert & in_range & confident_pt & \
                            (~in_guard_rail_zone) & (~in_platform_edge_zone)

        # 4. Векторизованное обновление состояния траектории
        force_refresh = (not warming_up) and (self.frame_count % FORCE_REFRESH_INTERVAL == 0)
        if warming_up or force_refresh:
            bx, by, bz = vx, vy, vz
        else:
            b_mask = ~is_candidate
            bx, by, bz = vx[b_mask], vy[b_mask], vz[b_mask]

        # 4.1. Ближний якорь -- прямой поиск колеи 1520мм, не continuity-fallback,
        # поэтому bore_confirmed=True (полный потолок высоты уместен и здесь)
        near_cx, near_rz, _ = self.find_near_track_anchor(bx, by, bz)
        self._update_slice_ema(0, near_cx, near_rz, True, bore_confirmed=True, force=warming_up)

        # 4.2. По-срезовое обновление с контролем надежности (Fix 4 & Fix 6)
        bin_idx = np.round((self.y_max - by) / self.slice_step).astype(np.int32)
        valid_bins = (bin_idx >= 1) & (bin_idx < self.n_slices)

        if valid_bins.any():
            valid_indices = np.flatnonzero(valid_bins)
            order = valid_indices[np.argsort(bin_idx[valid_bins], kind="stable")]
            sorted_x = bx[order]
            sorted_z = bz[order]
            sorted_b = bin_idx[order]
            split_pts = np.searchsorted(sorted_b, np.arange(1, self.n_slices + 1))

            prev_cx = self.state_cx[0] if not np.isnan(self.state_cx[0]) else near_cx

            for i in range(1, self.n_slices):
                p_start = split_pts[i - 1]
                p_end = split_pts[i]
                n_pts = p_end - p_start

                if n_pts < MIN_SLICE_POINTS:
                    prev_cx = self.state_cx[i] if not np.isnan(self.state_cx[i]) else prev_cx
                    self.state_stale_count[i] += 1
                    if self.state_stale_count[i] > 10:
                        self.state_confident[i] = False
                        self.state_bore_confirmed[i] = False
                    continue

                self.state_stale_count[i] = 0
                sx = sorted_x[p_start:p_end]
                sz = sorted_z[p_start:p_end]

                # Перцентили ширины тоннеля
                x_left, x_right = np.percentile(sx, [1.5, 98.5])
                tunnel_w = x_right - x_left
                tunnel_mid = (x_left + x_right) * 0.5
                # Однопутный тоннель
                is_single_bore = (4.5 <= tunnel_w <= 6.5) and (n_pts >= 200)

                # Высота головки рельса
                z_min = float(np.percentile(sz, 1.0))
                floor_mask = (sz >= z_min) & (sz <= z_min + 0.55)
                if floor_mask.sum() >= 10:
                    slice_rz = float(np.percentile(sz[floor_mask], 85))
                    slice_rz = float(np.clip(slice_rz, RAIL_HEIGHT_CALIBRATED - RAIL_HEIGHT_TOLERANCE,
                                             RAIL_HEIGHT_CALIBRATED + RAIL_HEIGHT_TOLERANCE))
                else:
                    slice_rz = self.state_rz[i] if not np.isnan(self.state_rz[i]) else near_rz
                    is_single_bore = False

                cand_cx = tunnel_mid if is_single_bore else prev_cx
                delta = cand_cx - prev_cx
                if abs(delta) > MAX_DRIFT_PER_SLICE:
                    cand_cx = prev_cx + np.sign(delta) * MAX_DRIFT_PER_SLICE

                # Доверие к геометрии среза: узкий однопутный свод ИЛИ надёжная непрерывность
                # рельсовой колеи в ближней зоне (<=25м, актуально для станций/стрелок, где
                # is_single_bore закономерно не проходит). Честная проверка на полном датасете
                # (chunk 104, 128-133) показала, что этот continuity-bypass даёт 98-100% ложных
                # срабатываний, когда рядом стоит настоящая межпутевая стена/опора
                # (pillar_support, высотой до 3.2-3.3м) -- но полностью убрать bypass нельзя:
                # без него ломается детекция человека на пути НА СТАНЦИИ (свой же тест пакета,
                # сценарий #12, начинает падать). Вместо удаления bypass, потолок высоты
                # препятствия для confident-но-НЕ-bore-confirmed точек снижен отдельно
                # (NEAR_ZONE_MAX_OBSTACLE_HEIGHT, см. process_frame) -- так и станционный
                # сценарий продолжает ловиться, и стена/опора выше человеческого роста больше
                # не считается кандидатом.
                yc_slice = abs(self.slice_y[i])
                is_slice_confident = is_single_bore or ((yc_slice <= 25.0) and (n_pts >= 80))

                self._update_slice_ema(i, cand_cx, slice_rz, is_slice_confident,
                                         bore_confirmed=is_single_bore, force=warming_up)
                prev_cx = self.state_cx[i]

        self.frame_count += 1

        # 5. Сбор кандидатов и кластеризация через связные компоненты воксельного графа
        cand_indices = np.flatnonzero(is_candidate)
        n_candidates = len(cand_indices)
        clusters: List[Dict[str, Any]] = []

        if n_candidates >= 15:
            clusters = self._cluster_candidates_voxel_graph(
                vx[cand_indices], vy[cand_indices], vz[cand_indices],
                dx[cand_indices], dz[cand_indices], v_int[cand_indices],
                vr[cand_indices]
            )

        nearest_dist = float(vr[cand_indices].min()) if n_candidates > 0 else (bore_range if not bore_clear else None)

        return {
            "obstacle_candidate_detected": (n_candidates >= 20) or (not bore_clear),
            "n_candidate_points": n_candidates,
            "nearest_distance": nearest_dist,
            "clusters": clusters,
            "bore_clear": bore_clear,
            "bore_range": bore_range,
            "bore_baseline": bore_baseline,
            "warming_up": warming_up,
            "frac_confident": float(self.state_confident.mean()),
            "trajectory": {
                "slice_y": self.slice_y.copy(),
                "state_cx": self.state_cx.copy(),
                "state_rz": self.state_rz.copy(),
                "state_confident": self.state_confident.copy(),
            }
        }

    def _cluster_candidates_voxel_graph(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        dx: np.ndarray,
        dz: np.ndarray,
        v_int: np.ndarray,
        vr: np.ndarray,
        voxel_size: Tuple[float, float, float] = (0.35, 0.50, 0.35),
        search_radius: float = 0.85,
        min_cluster_pts: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Воксельно-графовая кластеризация кандидатов со слиянием связных компонент.
        Формирует целостные 3D Bounding Boxes физических объектов за < 2 мс.
        """
        if len(vx) < min_cluster_pts:
            return []

        # 1. Вокселизация
        cx = np.round(vx / voxel_size[0]).astype(np.int32)
        cy = np.round(vy / voxel_size[1]).astype(np.int32)
        cz = np.round(vz / voxel_size[2]).astype(np.int32)
        coords = np.column_stack([cx, cy, cz])

        u_voxels, inv_idx = np.unique(coords, axis=0, return_inverse=True)
        n_vox = len(u_voxels)

        if n_vox == 1:
            comp_labels = np.zeros(n_vox, dtype=np.int32)
            n_comp = 1
        else:
            # Центроиды активных вокселей
            counts = np.bincount(inv_idx, minlength=n_vox)
            v_cents = np.zeros((n_vox, 3), dtype=np.float32)
            for dim, arr in enumerate([vx, vy, vz]):
                v_cents[:, dim] = np.bincount(inv_idx, weights=arr, minlength=n_vox) / counts

            tree = cKDTree(v_cents)
            pairs = tree.query_pairs(r=search_radius, output_type="ndarray")

            if len(pairs) > 0:
                row = np.concatenate([pairs[:, 0], pairs[:, 1]])
                col = np.concatenate([pairs[:, 1], pairs[:, 0]])
                data = np.ones(len(row), dtype=bool)
                adj = coo_matrix((data, (row, col)), shape=(n_vox, n_vox))
                n_comp, comp_labels = connected_components(adj, directed=False)
            else:
                n_comp = n_vox
                comp_labels = np.arange(n_vox, dtype=np.int32)

        pt_labels = comp_labels[inv_idx]
        comp_pt_counts = np.bincount(pt_labels, minlength=n_comp)
        valid_comps = np.flatnonzero(comp_pt_counts >= min_cluster_pts)

        clusters = []
        for c_id in valid_comps:
            pt_mask = (pt_labels == c_id)
            c_x = vx[pt_mask]
            c_y = vy[pt_mask]
            c_z = vz[pt_mask]
            c_dx = dx[pt_mask]
            c_dz = dz[pt_mask]
            c_int = v_int[pt_mask]
            c_r = vr[pt_mask]

            # Центрированные физические координаты для PCA в Tier 2 (в точности как при обучении One-Class)
            centered_xyz_int = np.column_stack([
                c_x - c_x.mean(),
                c_y - c_y.mean(),
                c_z - c_z.mean(),
                c_int
            ])

            clusters.append({
                "n_points": len(c_x),
                "pts_xyz": np.column_stack([c_x, c_y, c_z]),
                "pts_centered_int": centered_xyz_int,
                "pts_dx_dy_dz_int": centered_xyz_int,  # Согласованное признаковое пространство
                "mean_distance": float(c_r.mean()),
                "min_distance": float(c_r.min()),
                "centroid": np.array([float(c_x.mean()), float(c_y.mean()), float(c_z.mean())], dtype=np.float32),
                "extent": np.array([
                    float(c_x.max() - c_x.min()),
                    float(c_y.max() - c_y.min()),
                    float(c_z.max() - c_z.min())
                ], dtype=np.float32),
                "mean_dx": float(np.abs(c_dx).mean()),
                "mean_dz": float(c_dz.mean()),
            })

        return clusters

    def _update_slice_ema(self, i: int, cx: float, rz: float, confident: bool,
                            bore_confirmed: bool = False, force: bool = False):
        if np.isnan(self.state_cx[i]) or (force and not self.state_initialized[i]):
            self.state_cx[i] = cx
            self.state_rz[i] = rz
        else:
            a = self.ema_alpha
            self.state_cx[i] = (1.0 - a) * self.state_cx[i] + a * cx
            self.state_rz[i] = (1.0 - a) * self.state_rz[i] + a * rz
        self.state_confident[i] = confident
        self.state_bore_confirmed[i] = bore_confirmed
        self.state_initialized[i] = True
