/* A page-owned lease controls automatic following; reconnect never re-arms it. */
document.addEventListener('DOMContentLoaded', () => {
    const start = document.getElementById('follow-start');
    const stop = document.getElementById('follow-stop');
    const label = document.getElementById('follow-status');
    let token = null, starting = false, revision = 0, renewing = false, polling = false;
    let state = null, leaving = false;

    async function api(data, keepalive = false) {
        const abort = new AbortController();
        const timer = setTimeout(() => abort.abort(), 500);
        try {
            const response = await fetch('/api/follow/' + (data ? 'control' : 'status'), {
                method: data ? 'POST' : 'GET', cache: 'no-store', keepalive,
                headers: data ? {'Content-Type': 'application/json'} : {},
                body: data ? JSON.stringify(data) : undefined, signal: abort.signal
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || '自动追踪服务请求失败');
            return result;
        } finally { clearTimeout(timer); }
    }

    function render(message) {
        const busy = Boolean(starting || token || state?.active || state?.stopping);
        start.disabled = busy || !state?.ready;
        stop.disabled = !busy;
        start.setAttribute('aria-pressed', String(Boolean(token)));
        label.textContent = message || state?.message || '自动追踪未启用';
        if (!message && state?.distance != null && state?.active) {
            label.textContent += ` · 距目标 ${state.distance.toFixed(2)} 米`;
        }
        window.dispatchEvent(new CustomEvent('auto-follow-state', {
            detail: {active: Boolean(starting || token || state?.active), stopping: Boolean(state?.stopping)}
        }));
    }

    async function halt(message = '正在停止自动追踪…', all = false) {
        const previous = token, pending = starting;
        token = null; starting = false; revision++;
        render(message);
        if (previous || pending || all) {
            try {
                const data = {action: 'stop'};
                if (previous && !all) data.token = previous;
                await api(data, true);
            } catch (error) { label.textContent = '停止未确认：' + error.message; }
        }
    }

    start.addEventListener('click', async () => {
        if (starting || token || document.hidden || leaving) return;
        starting = true;
        const version = ++revision;
        const handoff = {waits: []};
        window.dispatchEvent(new CustomEvent('auto-follow-request', {detail: handoff}));
        render('正在接管导航与云台…');
        try {
            await Promise.all(handoff.waits);
            if (version !== revision || document.hidden || leaving) return;
            const result = await api({action: 'start'});
            if (version !== revision || document.hidden || leaving) {
                await api({action: 'stop', token: result.token}, true);
                return;
            }
            token = result.token;
            revision++; // Discard status requests sent before ownership was acquired.
            starting = false;
            render('自动追踪已启用');
        } catch (error) {
            if (version === revision) {
                starting = false;
                render(error.message);
            }
        }
    });
    stop.addEventListener('click', () => halt('正在取消导航…', true));
    window.addEventListener('manual-control-request', event => {
        if (token || starting) event.detail.waits.push(halt('正在切换到键盘驾驶…'));
    });
    window.addEventListener('manual-control-ready', () => {
        token = null; starting = false; revision++;
        if (state) state = {...state, active: false, stopping: false, message: '已切换到键盘驾驶'};
        render();
    });
    document.getElementById('teleop-stop').addEventListener('click', () => {
        if (token || starting || state?.active) halt('正在取消自动追踪…', true);
    });

    async function heartbeat() {
        if (!token || renewing || leaving || document.hidden) return;
        const session = token;
        renewing = true;
        try { await api({action: 'heartbeat', token: session}); }
        catch (error) { if (token === session) await halt(error.message); }
        finally { renewing = false; }
    }
    async function poll() {
        if (polling || leaving) return;
        polling = true;
        const version = revision;
        try {
            const result = await api();
            if (version !== revision) return;
            state = result;
            if (token && !state.active) { token = null; revision++; }
            render();
        } catch (error) {
            if (version !== revision) return;
            state = null;
            await halt('自动追踪服务连接中断');
        } finally { polling = false; }
    }
    window.addEventListener('blur', () => { if (token || starting) halt(); });
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && (token || starting)) halt();
    });
    window.addEventListener('pagehide', () => { leaving = true; halt(); });
    window.addEventListener('pageshow', () => { leaving = false; poll(); });
    document.addEventListener('keydown', event => {
        if ((event.code === 'Escape' || event.code === 'Space') && (token || starting || state?.active)) {
            event.preventDefault(); halt('正在停止自动追踪…', true);
        }
    }, true);
    document.getElementById('gimbal-stop').addEventListener('click', () => {
        if (token || starting || state?.active) halt('停止云台，自动追踪结束', true);
    });
    setInterval(heartbeat, 200);
    setInterval(poll, 300);
    poll();
});
