#!/bin/bash
# Launch ROS Bridge and Web App

# Source ROS 2
source /opt/ros/humble/setup.bash
source ~/nav_ws/install/setup.bash

# Trap Ctrl+C to kill background processes
trap "kill 0" EXIT

echo "Starting ROS Bridge Server..."
ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
ROSBRIDGE_PID=$!

# 启动 NoVNC (连接现有 RViz)
echo "启动 NoVNC 服务..."
./launch_rviz_web.sh &

# 启动 Web App
echo "正在启动 Flask Web App..."
python3 web_app.py

wait
