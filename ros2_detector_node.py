#!/usr/bin/env python3
"""
ROS 2 Humble Perception Node (Project 14 - Metro Moscow / Autonomous Train)
Подписывается на топик лидара PointCloud2, выполняет гибридную детекцию
препятствий в реальном времени и публикует флаги безопасности и 3D-маркеры в RViz2.
"""
import sys
import os
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Bool, Float32, String
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_clearance_solution.hybrid_pipeline import HybridPerceptionPipeline

DTYPE_MAP = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pointcloud2_to_numpy(msg: "PointCloud2"):
    """Быстрая конвертация ROS 2 PointCloud2 в именованный массив numpy."""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        names.append(f.name)
        formats.append(DTYPE_MAP[f.datatype])
        offsets.append(f.offset)
    dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    n = msg.width * msg.height
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    return arr


if ROS2_AVAILABLE:
    class MetroLiDARDetectorNode(Node):
        def __init__(self):
            super().__init__("metro_lidar_detector_node")

            # Параметры узла
            self.declare_parameter("lidar_topic", "/hesai/pandar_points")
            self.declare_parameter("frame_id", "hesai_lidar")
            self.declare_parameter("y_max", -2.5)
            self.declare_parameter("y_min", -60.0)

            lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
            self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

            self.get_logger().info(f"Инициализация детектора препятствий метро на топике: {lidar_topic}")

            # Инициализация конвейера
            self.pipeline = HybridPerceptionPipeline(
                y_max=self.get_parameter("y_max").get_parameter_value().double_value,
                y_min=self.get_parameter("y_min").get_parameter_value().double_value,
            )

            # Издатели (Publishers) по ТЗ хакатона
            self.pub_obstacle = self.create_publisher(Bool, "/metro/obstacle_detected", 10)
            self.pub_distance = self.create_publisher(Float32, "/metro/obstacle_distance", 10)
            self.pub_threat = self.create_publisher(String, "/metro/threat_level", 10)
            self.pub_markers = self.create_publisher(MarkerArray, "/metro/obstacle_markers", 10)
            self.pub_envelope = self.create_publisher(Marker, "/metro/clearance_envelope", 10)

            # Подписчик (Subscriber)
            self.sub_lidar = self.create_subscription(
                PointCloud2,
                lidar_topic,
                self.pointcloud_callback,
                10
            )

            self.get_logger().info("Узел детекции препятствий успешно запущен и готов к работе.")

        def pointcloud_callback(self, msg: PointCloud2):
            try:
                arr = pointcloud2_to_numpy(msg)
                x = arr["x"].astype(np.float32)
                y = arr["y"].astype(np.float32)
                z = arr["z"].astype(np.float32)
                intens = arr["intensity"].astype(np.float32) if "intensity" in arr.dtype.names else None
                ring = arr["ring"].astype(np.int32) if "ring" in arr.dtype.names else None

                timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

                # Обработка через оптимизированный гибридный конвейер
                res = self.pipeline.process_point_cloud(
                    x, y, z, intensity=intens, ring=ring, timestamp=timestamp
                )

                # 1. Публикация флага обнаружения
                msg_obs = Bool()
                msg_obs.data = bool(res["obstacle_detected"])
                self.pub_obstacle.publish(msg_obs)

                # 2. Публикация дистанции до препятствия
                msg_dist = Float32()
                msg_dist.data = float(res["obstacle_distance"]) if res["obstacle_distance"] is not None else -1.0
                self.pub_distance.publish(msg_dist)

                # 3. Публикация уровня угрозы
                msg_threat = String()
                msg_threat.data = f"{res['threat_level']}:{res['hazard_type']}"
                self.pub_threat.publish(msg_threat)

                # 4. Публикация маркеров препятствий для RViz2
                self.publish_rviz_markers(res["confirmed_boxes"], msg.header.stamp)

                # Логирование при срабатывании
                if res["obstacle_detected"]:
                    self.get_logger().warn(
                        f"ПРЕПЯТСТВИЕ НА ПУТИ! Дистанция: {res['obstacle_distance']:.1f} м | "
                        f"Тип: {res['hazard_type']} | Задержка: {res['latency_ms']:.1f} мс ({res['fps']:.1f} FPS)"
                    )
            except Exception as e:
                self.get_logger().error(f"Ошибка обработки кадра лидара: {e}")

        def publish_rviz_markers(self, boxes, stamp):
            marker_array = MarkerArray()

            # Удаление старых маркеров
            del_marker = Marker()
            del_marker.action = Marker.DELETEALL
            marker_array.markers.append(del_marker)

            for idx, b in enumerate(boxes):
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = stamp
                m.ns = "metro_obstacles"
                m.id = idx + 1
                m.type = Marker.CUBE
                m.action = Marker.ADD

                m.pose.position.x = float(b["x"])
                m.pose.position.y = float(b["y"])
                m.pose.position.z = float(b["z"])
                m.pose.orientation.w = 1.0

                m.scale.x = float(b["size_x"])
                m.scale.y = float(b["size_y"])
                m.scale.z = float(b["size_z"])

                # Красный цвет для опасных объектов
                m.color.r = 1.0
                m.color.g = 0.1
                m.color.b = 0.1
                m.color.a = 0.75

                marker_array.markers.append(m)

            self.pub_markers.publish(marker_array)


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS 2 (rclpy) не установлен в текущем окружении. Запустите в Docker-контейнере ROS 2 Humble.")
        return

    rclpy.init(args=args)
    node = MetroLiDARDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
