// Fake HTTP service; no connections to ROS, camera or hardware.
const intervals = new Map(); let timerId = 0;
window.setInterval = (callback, period) => { intervals.set(++timerId, {callback, period}); return timerId; };
window.clearInterval = id => intervals.delete(id);
const commands = [], states = [];
let feedback = {active: false, stopping: false, ready: true, distance: null, standoff: 1, message: '未启用'};
let currentToken = null, tokenNumber = 0, failHeartbeat = false, deferStart = false, finishStart;
let deferStatus = false, rejectStatus;
window.addEventListener('auto-follow-state', event => states.push(event.detail));
window.fetch = async (url, options) => {
    const reply = (data, ok = true) => ({ok, json: async () => data});
    if (url.endsWith('/status')) {
        if (deferStatus) return new Promise((resolve, reject) => { rejectStatus = reject; });
        return reply({...feedback});
    }
    const data = JSON.parse(options.body); commands.push(data);
    if (data.action === 'start') {
        feedback.active = true; currentToken = 'session-' + (++tokenNumber);
        const result = reply({token: currentToken});
        if (deferStart) return new Promise(resolve => { finishStart = () => resolve(result); });
        return result;
    }
    if (data.action === 'stop') {
        if (!data.token || data.token === currentToken) { feedback.active = false; currentToken = null; }
        return reply({ok: true});
    }
    if (data.action === 'heartbeat' && failHeartbeat) throw new Error('network failure');
    return reply({ok: true});
};
window.addEventListener('load', async () => {
    const passed = [], failures = [];
    const start = document.getElementById('follow-start'), stop = document.getElementById('follow-stop');
    const label = document.getElementById('follow-status');
    const assert = (value, message) => { if (!value) throw new Error(message); };
    const settle = async () => { for (let i=0; i<40; i++) await Promise.resolve(); };
    const click = async element => { element.click(); await settle(); };
    const tick = async period => { intervals.forEach(t => { if (t.period===period) t.callback(); }); await settle(); };
    const active = () => start.getAttribute('aria-pressed')==='true';
    const check = async (name, fn) => { try { await fn(); passed.push(name); } catch(e) { failures.push(name+': '+e.message); } };
    await settle();
    await check('viewing page never starts automatic motion', async () => {
        assert(commands.length===0 && !start.disabled && stop.disabled, 'initial state');
    });
    await check('start waits for gimbal handoff and locks manual mode', async () => {
        let release;
        const handoff = e => e.detail.waits.push(new Promise(resolve => {release=resolve;}));
        window.addEventListener('auto-follow-request', handoff);
        start.click(); await settle();
        assert(commands.length===0 && states.at(-1).active, 'did not wait for handoff');
        release(); await settle(); window.removeEventListener('auto-follow-request', handoff);
        assert(active() && start.disabled && !stop.disabled, 'not started');
    });
    await check('heartbeat owns a session and displays one metre status', async () => {
        await tick(200); assert(commands.at(-1).action==='heartbeat' && commands.at(-1).token===currentToken, 'lease');
        feedback.distance = 1.; await tick(300);
        assert(label.textContent.includes('1.00'), 'distance');
    });
    await check('explicit stop cancels and ends renewals', async () => {
        await click(stop); const count=commands.length; await tick(200);
        assert(!active() && commands.length===count && !feedback.active, 'stopped');
        await tick(300);
    });
    await check('blur stops and focus does not resume', async () => {
        await click(start); window.dispatchEvent(new Event('blur')); await settle(); await tick(300);
        assert(!active() && !feedback.active, 'blur');
        window.dispatchEvent(new Event('focus')); await settle(); assert(!active(), 'resumed');
    });
    await check('late start response after blur is canceled', async () => {
        deferStart=true; start.click(); await settle();
        window.dispatchEvent(new Event('blur')); await settle(); finishStart(); await settle(); deferStart=false;
        assert(!active() && !feedback.active && commands.at(-1).action==='stop', 'late response resumed');
        await tick(300);
    });
    await check('heartbeat failure stops and requires new click', async () => {
        await click(start); failHeartbeat=true; await tick(200); failHeartbeat=false;
        assert(!active() && !feedback.active, 'failure ignored'); await tick(300);
    });
    await check('old status failure cannot stop a newly started session', async () => {
        deferStatus=true; await tick(300);
        await click(start);
        assert(active(), 'start failed');
        rejectStatus(new Error('old request timed out')); await settle(); deferStatus=false;
        assert(active() && feedback.active, 'old request stopped new session');
        await click(stop); await tick(300);
    });
    await check('target loss reported by server releases controls', async () => {
        await click(start); feedback.active=false; feedback.message='目标已丢失'; await tick(300);
        assert(!active() && !states.at(-1).active && label.textContent==='目标已丢失', 'lost target');
    });
    await check('space stops even when a different page owns follow', async () => {
        feedback.active=true; await tick(300);
        document.dispatchEvent(new KeyboardEvent('keydown', {code:'Space', bubbles:true})); await settle();
        assert(!feedback.active && !active(), 'space ignored'); await tick(300);
    });
    await check('hidden page stops without restoring session', async () => {
        await click(start); Object.defineProperty(document,'hidden',{value:true, configurable:true});
        document.dispatchEvent(new Event('visibilitychange')); await settle(); delete document.hidden;
        window.dispatchEvent(new Event('pageshow')); await settle();
        assert(!active() && !feedback.active, 'visibility');
    });
    const report=document.createElement('pre'); report.id='test-results'; report.hidden=true;
    report.textContent=JSON.stringify({passed,failures}); document.body.append(report);
});
