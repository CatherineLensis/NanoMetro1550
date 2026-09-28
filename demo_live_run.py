#!/usr/bin/env python3
"""
Демо-скрипт для записи видео работы решения (см. README, раздел "Демо-видео").
Использует ТОТ ЖЕ пакет hybrid_clearance_solution, что и ros2_detector_node.py
(тот же HybridPerceptionPipeline) -- отличие только в источнике кадров: здесь
bag читается напрямую, а не через топик ROS 2, чтобы обойти известное
ограничение передачи крупных (~5МБ) PointCloud2 через DDS в виртуализированной
сети Docker Desktop на Windows (см. README, "Известные ограничения демо-записи").
На целевом стенде (нативный Ubuntu, п.3.1 ТЗ) эта проблема, как правило, не
возникает -- там для демонстрации можно использовать штатный
`ros2 bag play` + `ros2_detector_node.py` + RViz2/Foxglove напрямую.
"""
import argparse
import sqlite3
import sys
import time

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, "/app")
from hybrid_clearance_solution.hybrid_pipeline import HybridPerceptionPipeline

DTYPE_MAP = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pc2_to_array(msg):
    names, formats, offsets = [], [], []
    for f in msg.fields:
        names.append(f.name); formats.append(DTYPE_MAP[f.datatype]); offsets.append(f.offset)
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    n = msg.width * msg.height
    return np.frombuffer(bytes(msg.data), dtype=dt, count=n)


def main():
    ap = argparse.ArgumentParser(description="Живое проигрывание bag с выводом детекции в консоль")
    ap.add_argument("bag_db3", help="путь к .db3 файлу бага")
    ap.add_argument("--rate", type=float, default=1.0, help="скорость проигрывания (1.0 = реальное время)")
    ap.add_argument("--start-frame", type=int, default=0,
                     help="с какого кадра НАЧАТЬ ПЕЧАТЬ (детектор обрабатывает ВСЕ кадры с 0 -- "
                          "он stateful, пропуск обработки исказил бы поведение)")
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--fast-forward-rate", type=float, default=None,
                     help="скорость проигрывания ДО --start-frame (по умолчанию = --rate, "
                          "можно поставить выше, чтобы быстрее промотать разогрев без искажения state)")
    args = ap.parse_args()
    ff_rate = args.fast_forward_rate or args.rate

    pipeline = HybridPerceptionPipeline()
    con = sqlite3.connect(f"file:{args.bag_db3}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT data FROM messages ORDER BY id;")

    print(f"=== Project 14: hybrid_clearance_solution — живое проигрывание {args.bag_db3} ===")
    print(f"=== Скорость: x{args.rate}, реальный интервал между кадрами лидара ~0.1с ===\n")

    if args.start_frame > 0:
        print(f"(разгоняю детектор через кадры 0-{args.start_frame - 1} на x{ff_rate} "
              f"без пропуска обработки -- Tier1 stateful, состояние должно быть честным)")

    fi = 0
    t_prev = time.perf_counter()
    for (data,) in cur:
        if args.end_frame is not None and fi > args.end_frame:
            break

        msg = deserialize_message(bytes(data), PointCloud2)
        pts = pc2_to_array(msg)
        x = pts["x"].astype(np.float32)
        y = pts["y"].astype(np.float32)
        z = pts["z"].astype(np.float32)
        intensity = pts["intensity"].astype(np.float32) if "intensity" in pts.dtype.names else np.zeros_like(x)

        out = pipeline.process_point_cloud(x, y, z, intensity=intensity)

        if fi >= args.start_frame:
            if out["obstacle_detected"]:
                print(f"[КАДР {fi:5d}] \033[91m ПРЕПЯТСТВИЕ НА ПУТИ! \033[0m "
                      f"Дистанция: {out['obstacle_distance']:.1f} м | Тип: {out['hazard_type']} | "
                      f"Задержка: {out['latency_ms']:.1f} мс ({out['fps']:.1f} FPS)")
            else:
                print(f"[КАДР {fi:5d}] чисто | {len(x)} точек | Задержка: {out['latency_ms']:.1f} мс", end="\r")

        # пейсинг: быстрая перемотка до start-frame, целевая скорость после
        rate = ff_rate if fi < args.start_frame else args.rate
        target_dt = 0.1 / rate
        elapsed = time.perf_counter() - t_prev
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)
        t_prev = time.perf_counter()
        fi += 1

    con.close()
    print("\n\n=== Проигрывание завершено ===")


if __name__ == "__main__":
    main()
