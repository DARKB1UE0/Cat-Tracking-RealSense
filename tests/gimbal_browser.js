// Replace Flask requests and intervals. No camera, USB or robot traffic.
const intervals = new Map();
let timerId = 0;
window.setInterval = (callback, period) => { intervals.set(++timerId, {callback, period}); return timerId; };
window.clearInterval = id => intervals.delete(id);
const commands = [];
let feedback = {connected: true, controllable: true, active: false,
    yaw: 12.5, pitch: -5, enabled: true, fault: 1, message: '云台已连接'};
let currentToken = null, tokenNumber = 0;
let failTarget = false, deferStart = false, finishStart = null;
window.fetch = async (url, options) => {
    const reply = (data, ok = true) => ({ok, json: async () => data});
    if (url.endsWith('/status')) return reply({...feedback});
    const data = JSON.parse(options.body);
    commands.push(data);
    if (data.action === 'start') {
        currentToken = 'session-' + (++tokenNumber);
        feedback.active = true;
        const result = reply({token: currentToken, yaw: feedback.yaw, pitch: feedback.pitch});
        if (deferStart) return new Promise(resolve => { finishStart = () => resolve(result); });
        return result;
    }
    if (data.action === 'stop') {
        if (!data.token || data.token === currentToken) { feedback.active = false; currentToken = null; }
        return reply({ok: true});
    }
    if (failTarget) throw new Error('网络中断');
    return reply({ok: true});
};
window.addEventListener('load', async () => {
    const passed = [], failures = [];
    const yaw = document.getElementById('gimbal-yaw');
    const pitch = document.getElementById('gimbal-pitch');
    const toggle = document.getElementById('gimbal-toggle');
    const home = document.getElementById('gimbal-home');
    const stop = document.getElementById('gimbal-stop');
    const assert = (value, message) => { if (!value) throw new Error(message); };
    const settle = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
    const targets = () => commands.filter(x => x.action === 'target');
    const active = () => toggle.getAttribute('aria-pressed') === 'true';
    const click = async element => { element.click(); await settle(); };
    const tick = async period => {
        intervals.forEach(timer => { if (timer.period === period) timer.callback(); });
        await settle();
    };
    const check = async (name, fn) => {
        try { await fn(); passed.push(name); } catch (e) { failures.push(name + ': ' + e.message); }
    };
    await settle();
    await check('initial page reads feedback without motion', async () => {
        assert(commands.length === 0 && yaw.disabled && pitch.disabled, 'unsolicited motion');
        assert(document.getElementById('gimbal-yaw-actual').textContent === '12.5°', 'actual angle');
        assert(!toggle.disabled, 'healthy feedback permits control');
    });
    await check('yaw exposes minus90 to plus90 and pitch minus30 to plus30', async () => {
        assert(yaw.min === '-90' && yaw.max === '90', 'yaw range');
        assert(pitch.min === '-30' && pitch.max === '30', 'pitch range');
    });
    await check('enable starts from measured angles', async () => {
        await click(toggle);
        assert(active() && !yaw.disabled && !pitch.disabled, 'not enabled');
        assert(targets().at(-1).yaw === 12.5 && targets().at(-1).pitch === -5, 'jumped to zero');
    });
    await check('mouse slider input sends signed targets and labels', async () => {
        yaw.value = '-90'; yaw.dispatchEvent(new Event('input')); await settle();
        pitch.value = '30'; pitch.dispatchEvent(new Event('input')); await settle();
        assert(targets().at(-1).yaw === -90 && targets().at(-1).pitch === 30, 'target axes');
        assert(document.getElementById('gimbal-pitch-value').textContent === '30.0°', 'label');
        assert(document.getElementById('gimbal-yaw-actual').textContent === '12.5°', 'target shown as measured');
    });
    await check('release keeps sending lease renewals', async () => {
        const count = targets().length;
        await tick(100); await tick(100);
        assert(targets().length === count + 2 && targets().at(-1).pitch === 30, 'hold missing');
        assert(targets().at(-1).sequence > targets().at(-2).sequence, 'sequence');
    });
    await check('one click home sends two zero targets and holds', async () => {
        await click(home); await tick(100);
        assert(targets().at(-1).yaw === 0 && targets().at(-1).pitch === 0, 'not home');
        assert(yaw.value === '0' && pitch.value === '0', 'sliders not home');
    });
    await check('stop invalidates local control and stops heartbeat', async () => {
        await click(stop);
        const count = targets().length; await tick(100);
        assert(!active() && yaw.disabled && targets().length === count, 'motion after stop');
        assert(commands.at(-1).action === 'stop', 'no stop frame request');
    });
    await check('one click home also works while control is disabled', async () => {
        await click(home);
        assert(active() && targets().at(-1).yaw === 0 && targets().at(-1).pitch === 0, 'home from idle');
    });
    await check('window blur stops and never automatically resumes', async () => {
        window.dispatchEvent(new Event('blur')); await settle(); await tick(250);
        assert(!active() && commands.at(-1).action === 'stop', 'blur');
        window.dispatchEvent(new Event('focus')); await tick(100);
        assert(!active(), 'automatic resume');
    });
    await check('motor fault disables controls', async () => {
        feedback.fault = 4; feedback.controllable = false; await tick(250);
        assert(toggle.disabled && home.disabled, 'fault ignored');
        assert(document.getElementById('gimbal-detail').textContent.includes('Roll 电机离线'), 'fault text');
        feedback.fault = 0; feedback.controllable = true; await tick(250);
    });
    await check('target network failure ends local control', async () => {
        await click(toggle); failTarget = true; await tick(100);
        assert(!active() && commands.at(-1).action === 'stop', 'network failure ignored');
        failTarget = false; await tick(250);
    });
    await check('late enable response after blur is stopped', async () => {
        deferStart = true; toggle.click(); await settle();
        window.dispatchEvent(new Event('blur')); await settle();
        finishStart(); await settle(); deferStart = false;
        assert(!active() && !feedback.active && commands.at(-1).action === 'stop', 'late enable resumed');
        await tick(250);
    });
    await check('other page ownership blocks enable but permits stop', async () => {
        feedback.active = true; await tick(250);
        assert(toggle.disabled && home.disabled && !stop.disabled, 'ownership');
        await click(stop); await tick(250);
    });
    await check('hidden page stops and page restoration stays idle', async () => {
        await click(toggle);
        Object.defineProperty(document, 'hidden', {value: true, configurable: true});
        document.dispatchEvent(new Event('visibilitychange')); await settle();
        assert(!active(), 'hidden control'); delete document.hidden;
        window.dispatchEvent(new Event('pagehide')); await settle();
        window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted: true})); await settle();
        assert(!active(), 'restored control');
    });
    const result = document.createElement('pre'); result.id = 'test-results'; result.hidden = true;
    result.textContent = JSON.stringify({passed, failures}); document.body.append(result);
});
