/* Mouse sliders -> Flask lease -> 50 Hz USB setpoints. */
document.addEventListener('DOMContentLoaded', () => {
    const yaw = document.getElementById('gimbal-yaw');
    const pitch = document.getElementById('gimbal-pitch');
    const toggle = document.getElementById('gimbal-toggle');
    const home = document.getElementById('gimbal-home');
    const stop = document.getElementById('gimbal-stop');
    const status = document.getElementById('gimbal-status');
    const detail = document.getElementById('gimbal-detail');
    let token = null, sequence = 0, revision = 0;
    let starting = false, sending = false, polling = false;
    let state = null, leaving = false;

    async function api(path, data) {
        const abort = new AbortController();
        const timeout = setTimeout(() => abort.abort(), 450);
        try {
            const response = await fetch('/api/gimbal/' + path, {
                method: data ? 'POST' : 'GET', cache: 'no-store',
                headers: data ? {'Content-Type': 'application/json'} : {},
                body: data ? JSON.stringify(data) : undefined,
                signal: abort.signal
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || '云台服务请求失败');
            return result;
        } finally {
            clearTimeout(timeout);
        }
    }

    function labels() {
        [yaw, pitch].forEach(input => {
            document.getElementById(input.id + '-value').textContent = Number(input.value).toFixed(1) + '°';
        });
    }

    function render(message) {
        const connected = Boolean(state && state.connected);
        const available = Boolean(state && state.controllable);
        const occupied = Boolean(state && state.active && !token);
        yaw.disabled = pitch.disabled = !token;
        toggle.disabled = starting || (!token && (!available || occupied));
        home.disabled = starting || (!token && (!available || occupied));
        stop.disabled = !connected && !token && !starting;
        toggle.textContent = token ? '关闭云台控制' : '启用云台控制';
        toggle.setAttribute('aria-pressed', String(Boolean(token)));
        status.classList.toggle('active', Boolean(token));
        status.textContent = message || (token ? '手动控制中 · 松开保持' :
            occupied ? '其他页面正在控制' : connected ? '云台已连接 · 控制未启用' : '云台未连接');
        labels();
    }

    function sendStop(session) {
        const data = {action: 'stop'};
        if (session) data.token = session;
        // Keepalive also works when the page is being hidden/unloaded.
        return fetch('/api/gimbal/control', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data), keepalive: true
        }).then(async response => {
            if (!response.ok) {
                const result = await response.json();
                throw new Error(result.error || '停止指令未确认');
            }
        });
    }

    function disable(message = '已停止云台', explicit = false) {
        const previous = token;
        const pending = starting;
        token = null;
        starting = false;
        revision++;
        if (state && previous) state.active = false;
        render(message);
        if (previous || pending || explicit) {
            sendStop(previous).catch(error => {
                status.textContent = '停止未确认 · 请检查连接';
                detail.textContent = error.message;
            });
        }
    }

    async function enable(returnHome = false) {
        if (starting || token || document.hidden || leaving) return;
        starting = true;
        const version = ++revision;
        render('正在启用云台…');
        try {
            const result = await api('control', {action: 'start'});
            if (version !== revision || document.hidden || leaving) {
                await sendStop(result.token);
                return;
            }
            token = result.token;
            sequence = 0;
            yaw.value = returnHome ? 0 : result.yaw;
            pitch.value = returnHome ? 0 : result.pitch;
            if (state) state.active = true;
            starting = false;
            render();
            await sendTarget();
        } catch (error) {
            if (version === revision) {
                starting = false;
                render(error.message);
            }
        }
    }

    async function sendTarget() {
        if (!token || sending || leaving || document.hidden) return;
        const session = token;
        sending = true;
        try {
            await api('control', {action: 'target', token: session, sequence: sequence++,
                yaw: Number(yaw.value), pitch: Number(pitch.value)});
        } catch (error) {
            if (token === session) disable(error.message);
        } finally {
            sending = false;
        }
    }

    async function poll() {
        if (polling || leaving) return;
        polling = true;
        const version = revision;
        try {
            const result = await api('status');
            if (version !== revision) return;
            state = result;
            [ ['yaw', result.yaw], ['pitch', result.pitch] ].forEach(([axis, value]) => {
                document.getElementById('gimbal-' + axis + '-actual').textContent =
                    result.connected && Number.isFinite(value) ? value.toFixed(1) + '°' : '—';
            });
            const faults = [ [2, 'Yaw 电机离线'], [4, 'Pitch 电机离线'],
                [8, 'Pitch 电机故障'], [16, '目标超限'], [32, '零位未就绪'] ]
                .filter(([bit]) => result.fault & bit).map(([, label]) => label);
            detail.textContent = !result.connected ? result.message : faults.length ? faults.join('；') :
                (result.enabled ? '电机已使能' : '电机已停止') +
                (result.fault & 1 ? ' · 无新目标，固件按默认策略保持' : '');
            if (token && (!result.controllable || !result.active)) disable('控制已结束，请重新启用');
            else render(faults.length ? '云台故障 · 请检查电机' : undefined);
        } catch (error) {
            if (version === revision) {
                state = null;
                disable('云台服务连接中断');
                detail.textContent = error.message;
                document.getElementById('gimbal-yaw-actual').textContent = '—';
                document.getElementById('gimbal-pitch-actual').textContent = '—';
            }
        } finally {
            polling = false;
        }
    }

    [yaw, pitch].forEach(input => input.addEventListener('input', () => {
        labels();
        sendTarget();
    }));
    toggle.addEventListener('click', () => token ? disable() : enable());
    home.addEventListener('click', () => {
        if (!token) enable(true);
        else { yaw.value = pitch.value = 0; labels(); sendTarget(); }
    });
    stop.addEventListener('click', () => disable('已停止云台', true));
    window.addEventListener('blur', () => { if (token || starting) disable(); });
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && (token || starting)) disable();
    });
    window.addEventListener('pagehide', () => { leaving = true; disable(); });
    window.addEventListener('pageshow', () => { leaving = false; poll(); });
    document.addEventListener('keydown', event => {
        if (event.code === 'Escape' && (token || starting)) disable();
    });
    setInterval(sendTarget, 100);
    setInterval(poll, 250);
    render('正在连接云台…');
    poll();
});
