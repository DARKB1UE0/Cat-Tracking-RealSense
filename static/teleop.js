/* Keyboard driving via the rosbridge server started by launch_web_nav.sh. */
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('teleop-toggle');
    const stopButton = document.getElementById('teleop-stop');
    const status = document.getElementById('teleop-status');
    const linear = document.getElementById('teleop-linear');
    const angular = document.getElementById('teleop-angular');
    const keyHints = document.querySelectorAll('[data-drive-key]');
    const bindings = {
        KeyW: [1, 0, 0], KeyS: [-1, 0, 0],
        KeyA: [0, 1, 0], KeyD: [0, -1, 0],
        KeyJ: [0, 0, 1], KeyK: [0, 0, -1]
    };
    const pressed = new Set();
    const topic = '/cmd_vel_manual';
    let autoBlocked = false;
    let starting = false, revision = 0;
    const bridgeUrl = new URL(window.location.href);
    bridgeUrl.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    bridgeUrl.port = '9090';
    bridgeUrl.pathname = '/';
    bridgeUrl.search = '';
    bridgeUrl.hash = '';
    let socket = null;
    let enabled = false;
    let publishTimer = null;
    let reconnectTimer = null;
    let leaving = false;

    function connected() {
        return socket !== null && socket.readyState === WebSocket.OPEN;
    }

    function updateControls(message) {
        toggle.disabled = !connected() || starting;
        stopButton.disabled = !connected();
        toggle.textContent = enabled ? '关闭键盘驾驶' : '启用键盘驾驶';
        toggle.setAttribute('aria-pressed', String(enabled));
        status.textContent = message || (starting ? '正在取消导航，等待键盘接管…' :
            enabled ? '键盘驾驶已启用' : autoBlocked ? '导航控制中 · 点击启用键盘可接管' : '通信已连接 · 驾驶未启用');
        status.classList.toggle('active', enabled);
        keyHints.forEach(hint => {
            hint.classList.toggle('pressed', pressed.has(hint.dataset.driveKey));
        });
    }

    function publish(x = 0, y = 0, z = 0) {
        // Never queue motion commands for a disconnected or congested socket.
        if (!connected()) return;
        if (socket.bufferedAmount > 0) {
            disable(false);
            updateControls('通信拥塞 · 驾驶已关闭');
            socket.close();
            return;
        }
        socket.send(JSON.stringify({
            op: 'publish', topic,
            msg: {
                linear: {x, y, z: 0},
                angular: {x: 0, y: 0, z}
            }
        }));
    }

    function clearKeys() {
        pressed.clear();
        clearInterval(publishTimer);
        publishTimer = null;
    }

    function disable(sendStop = true) {
        const wasEnabled = enabled;
        revision++;
        starting = false;
        enabled = false;
        clearKeys();
        if (sendStop && wasEnabled) publish();
        updateControls();
    }

    function publishMotion() {
        if (!enabled || !connected() || document.hidden) {
            disable();
            return;
        }
        let x = 0, y = 0, z = 0;
        pressed.forEach(key => {
            const direction = bindings[key];
            x += direction[0];
            y += direction[1];
            z += direction[2];
        });
        // Keep diagonal movement at the selected translation speed.
        const scale = Number(linear.value) / Math.max(1, Math.hypot(x, y));
        publish(x * scale, y * scale, z * Number(angular.value));
    }

    function connect() {
        if (leaving) return;
        updateControls('正在连接底盘通信…');
        socket = new WebSocket(bridgeUrl.href);
        socket.addEventListener('open', () => {
            socket.send(JSON.stringify({
                op: 'advertise', topic, type: 'geometry_msgs/msg/Twist',
                queue_size: 1, latch: false
            }));
            // Reconnection must never restore held keys or enable driving.
            disable(false);
        });
        socket.addEventListener('close', () => {
            disable(false);
            updateControls('通信已断开 · 等待重连');
            if (!leaving) reconnectTimer = setTimeout(connect, 2000);
        });
        socket.addEventListener('error', () => {
            disable(false);
            updateControls('连接失败 · 请检查网页启动服务');
            socket.close();
        });
        socket.addEventListener('message', event => {
            let message;
            try { message = JSON.parse(event.data); } catch { return; }
            if (message.op === 'status' && message.level === 'error') {
                disable();
                updateControls('底盘通信报错 · 请检查 ROS 服务');
                socket.close();
            }
        });
    }

    function isEditing(target) {
        return target instanceof Element && (
            target.isContentEditable || target.closest('input, textarea, select, [role="textbox"]')
        );
    }

    toggle.addEventListener('click', async () => {
        if (enabled) disable();
        else if (connected() && !starting && !document.hidden && !leaving) {
            starting = true;
            const version = ++revision;
            const handoff = {waits: []};
            window.dispatchEvent(new CustomEvent('manual-control-request', {detail: handoff}));
            updateControls();
            const abort = new AbortController();
            const timer = setTimeout(() => abort.abort(), 8000);
            try {
                await Promise.all(handoff.waits);
                if (version !== revision) return;
                const response = await fetch('/api/teleop/enable', {method: 'POST', signal: abort.signal});
                const result = await response.json();
                if (!response.ok) throw new Error(result.error || '导航取消失败');
                if (version !== revision || !connected() || document.hidden || leaving) return;
                starting = false;
                autoBlocked = false;
                window.dispatchEvent(new Event('manual-control-ready'));
                enabled = true;
                updateControls();
            } catch (error) {
                if (version === revision) {
                    disable();
                    updateControls('键盘未接管 · ' + error.message);
                }
            } finally { clearTimeout(timer); }
        }
        // Keep subsequent Space presses from activating this button again.
        toggle.blur();
    });
    stopButton.addEventListener('click', () => {
        disable(false);
        publish();
        stopButton.blur();
    });

    document.addEventListener('keydown', event => {
        if (starting && (event.code === 'Space' || event.code === 'Escape')) {
            event.preventDefault(); disable(); return;
        }
        if (!enabled) return;
        if (event.ctrlKey || event.altKey || event.metaKey || event.isComposing || isEditing(event.target)) {
            disable();
            return;
        }
        if (event.code === 'Space' || event.code === 'Escape') {
            event.preventDefault();
            disable();
            return;
        }
        if (!bindings[event.code]) return;
        event.preventDefault();
        if (event.repeat || pressed.has(event.code)) return;
        pressed.add(event.code);
        publishMotion();
        if (enabled && publishTimer === null) publishTimer = setInterval(publishMotion, 50);
        updateControls();
    });
    document.addEventListener('keyup', event => {
        if (!pressed.delete(event.code)) return;
        event.preventDefault();
        publishMotion();
        if (pressed.size === 0) clearKeys();
        updateControls();
    });
    document.addEventListener('focusin', event => {
        if (isEditing(event.target) || event.target.tagName === 'IFRAME') disable();
    });
    window.addEventListener('blur', () => disable());
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) disable();
    });
    window.addEventListener('pagehide', () => {
        leaving = true;
        clearTimeout(reconnectTimer);
        disable();
        if (socket) socket.close();
    });
    window.addEventListener('pageshow', event => {
        if (event.persisted) {
            leaving = false;
            connect();
        }
    });
    [[linear, 'm/s'], [angular, 'rad/s']].forEach(([input, unit]) => {
        input.addEventListener('input', () => {
            document.getElementById(`${input.id}-value`).textContent = `${Number(input.value).toFixed(2)} ${unit}`;
        });
    });
    window.addEventListener('auto-follow-state', event => {
        autoBlocked = event.detail.active || event.detail.stopping;
        if (autoBlocked && !starting) disable();
        updateControls();
    });
    window.addEventListener('auto-follow-request', () => disable());
    connect();
});
