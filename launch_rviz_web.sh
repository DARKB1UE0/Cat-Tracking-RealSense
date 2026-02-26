#!/bin/bash

# Configuration
DISPLAY_NUM=":0" # Use physical display
VNC_PORT=5900
WEB_PORT=6080
NOVNC_DIR="/home/bigtruck/nav_ws/src/Cat-Tracking-RealSense/static/novnc"

# Kill previous instances
pkill -f x11vnc
pkill -f websockify

# Start websockify immediately
echo "Starting Websockify on port $WEB_PORT..."
websockify --web $NOVNC_DIR $WEB_PORT localhost:$VNC_PORT &
WEBSOCKIFY_PID=$!
echo "Websockify started with PID $WEBSOCKIFY_PID"

export DISPLAY=$DISPLAY_NUM

echo "Searching for RViz window on $DISPLAY_NUM..."
echo "Please open RViz if you haven't already!"

# Loop indefinitely until RViz is found
# Match window title ending with "- RViz" (e.g. "navigation.rviz* - RViz")
# This avoids matching editor windows that have "rviz" in file paths
RVIZ_WIN_ID=""
while [ -z "$RVIZ_WIN_ID" ]; do
    RVIZ_WIN_ID=$(wmctrl -l | grep -- "- RViz" | awk '{print $1}' | head -n 1)
    
    if [ -z "$RVIZ_WIN_ID" ]; then
        echo "Waiting for RViz window... (Open RViz to start VNC stream)"
        sleep 3
    fi
done

echo "Found RViz Window ID: $RVIZ_WIN_ID"

# Force RViz window to be Always on Top
echo "Setting RViz window to 'Always on Top'..."
wmctrl -i -r $RVIZ_WIN_ID -b add,above

echo "Starting x11vnc for single window..."

# Start x11vnc attached to RViz window with performance tuning
# XDG_SESSION_TYPE=x11 and unset WAYLAND_DISPLAY: bypass Wayland detection
# RViz runs via XWayland so x11vnc can capture it through X11 protocol
unset WAYLAND_DISPLAY
export XDG_SESSION_TYPE=x11
x11vnc -display $DISPLAY_NUM \
       -id $RVIZ_WIN_ID \
       -forever \
       -shared \
       -rfbport $VNC_PORT \
       -bg \
       -o /tmp/x11vnc.log \
       -noxdamage \
       -noshm \
       -repeat

echo "VNC Server running on port $VNC_PORT"

wait $WEBSOCKIFY_PID
