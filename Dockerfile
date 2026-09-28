# Dockerfile для решения кейса беспилотного поезда метро (Project 14)
# Базовый образ: ROS 2 Humble на Ubuntu 22.04 LTS
FROM ros:humble-ros-base-jammy

LABEL maintainer="Project 14 Team"
LABEL description="Autonomous Metro LiDAR Obstacle Detection Pipeline"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей и инструментов визуализации
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-pandas \
    python3-sklearn \
    python3-matplotlib \
    ros-humble-rviz2 \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-sensor-msgs \
    ros-humble-visualization-msgs \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование исходных кодов и обученных моделей
COPY . /app/hybrid_clearance_solution

ENV PYTHONPATH="/app"

# Скрипт запуска по умолчанию
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 -m hybrid_clearance_solution.ros2_detector_node"]
