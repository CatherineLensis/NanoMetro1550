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

Опционально (--live-viz-dir) пишет текущий кадр облака точек как PNG в
указанную папку в реальном времени -- см. viewer.html рядом с этим файлом:
откройте его в браузере, указав ту же папку, для живой визуализации, идущей
синхронно с логом в терминале (без RViz2/Foxglove, требование ТЗ п.4 про
демонстрацию в реальном времени).
"""
import argparse
import os
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
    ap.add_argument("--live-viz-dir", type=str, default=None,
                     help="папка (обычно смонтированная с хоста), куда писать live.png на каждом кадре "
                          ">= --start-frame -- открой viewer.html из той же папки в браузере")
    args = ap.parse_args()
    ff_rate = args.fast_forward_rate or args.rate

    live_viz = None
    if args.live_viz_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.live_viz_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 9), facecolor="white")
        live_viz = (plt, fig, ax)
        # кладём viewer.html рядом, если его там ещё нет
        viewer_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
        viewer_dst = os.path.join(args.live_viz_dir, "viewer.html")
        if os.path.exists(viewer_src) and not os.path.exists(viewer_dst):
            with open(viewer_src, "r", encoding="utf-8") as f_in, open(viewer_dst, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())

    pipeline = HybridPerceptionPipeline()
    con = sqlite3.connect(f"file:{args.bag_db3}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT data FROM messages ORDER BY id;")

    print(f"=== Project 14: hybrid_clearance_solution — живое проигрывание {args.bag_db3} ===")
    print(f"=== Скорость: x{args.rate}, реальный интервал между кадрами лидара ~0.1с ===")
    if live_viz:
        print(f"=== Живая визуализация: {os.path.join(args.live_viz_dir, 'live.png')} "
              f"(открой viewer.html в браузере) ===")
    print()

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

            if live_viz:
                plt_mod, fig, ax = live_viz
                ax.clear()
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & ((x*x + y*y + z*z) > 0.09)
                bg_idx = np.flatnonzero(valid)[::6]
                ax.scatter(x[bg_idx], y[bg_idx], s=1.5, c="#B0B0B0", linewidths=0)
                for b in out["confirmed_boxes"]:
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((b["x"] - b["size_x"] / 2, b["y"] - b["size_y"] / 2),
                                      b["size_x"], b["size_y"], linewidth=2,
                                      edgecolor="#E8383D", facecolor="#E8383D", alpha=0.35)
                    ax.add_patch(rect)
                ax.set_xlim(-6, 6); ax.set_ylim(-40, 2)
                ax.set_xlabel("X, м (поперёк)"); ax.set_ylabel("Y, м (вдоль пути)")
                status = (f"ОБНАРУЖЕНО: {out['hazard_type']} @ {out['obstacle_distance']:.1f} м"
                          if out["obstacle_detected"] else "чисто")
                color = "#B00020" if out["obstacle_detected"] else "#1B7A3E"
                ax.set_title(f"кадр {fi}\n{status}", fontsize=12, color=color, fontweight="bold")
                ax.set_aspect("equal")
                tmp_path = os.path.join(args.live_viz_dir, "live.png.tmp")
                final_path = os.path.join(args.live_viz_dir, "live.png")
                fig.savefig(tmp_path, dpi=100, format="png")
                os.replace(tmp_path, final_path)  # атомарная замена -- вьюер не увидит "половину" файла

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
