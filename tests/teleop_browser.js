// This fake replaces WebSocket before application code loads. No ROS connection is made.
const sockets = [];
const intervals = new Map();
const retries = new Map();
let timerId = 0;
window.setInterval = callback => { intervals.set(++timerId, callback); return timerId; };
window.clearInterval = id => intervals.delete(id);
window.setTimeout = callback => { retries.set(++timerId, callback); return timerId; };
window.clearTimeout = id => retries.delete(id);
window.WebSocket = class extends EventTarget {
    static OPEN = 1;
    constructor(url) {
        super();
        this.url = url;
        this.readyState = 0;
        this.bufferedAmount = 0;
        this.sent = [];
        sockets.push(this);
    }
    open() { this.readyState = 1; this.dispatchEvent(new Event('open')); }
    send(data) {
        if (this.readyState !== 1) throw new Error('send on closed socket');
        this.sent.push(JSON.parse(data));
    }
    close() {
        if (this.readyState === 3) return;
        this.readyState = 3;
        this.dispatchEvent(new Event('close'));
    }
};

// load fires after the application's DOMContentLoaded handler.
window.addEventListener('load', () => {
    const passed = [], failures = [];
    const toggle = document.getElementById('teleop-toggle');
    const stop = document.getElementById('teleop-stop');
    const current = () => sockets[sockets.length - 1];
    const publications = () => current().sent.filter(message => message.op === 'publish');
    const last = () => publications().at(-1).msg;
    const assert = (value, message) => { if (!value) throw new Error(message); };
    const check = (name, run) => {
        try { run(); passed.push(name); } catch (error) { failures.push(name + ': ' + error.message); }
    };
    const key = (code, type = 'keydown', options = {}, target = document.body) => {
        const event = new KeyboardEvent(type, {code, bubbles: true, cancelable: true, ...options});
        target.dispatchEvent(event);
        return event;
    };
    const active = () => toggle.getAttribute('aria-pressed') === 'true';
    const enable = () => { if (!active()) toggle.click(); assert(active(), 'must enable'); };
    const zero = () => {
        const msg = last();
        assert([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z]
            .every(value => value === 0), 'expected a zero Twist');
    };

    check('disconnected controls disabled', () => assert(toggle.disabled && stop.disabled, 'buttons'));
    check('connect to page host on port 9090', () => {
        assert(current().url === 'ws://127.0.0.1:9090/', current().url);
        current().open();
        const ad = current().sent[0];
        assert(ad.op === 'advertise' && ad.topic === '/cmd_vel' && ad.type === 'geometry_msgs/msg/Twist', 'advertisement');
        assert(!toggle.disabled && !active(), 'connected but disabled');
    });
    check('idle page does not publish', () => {
        key('KeyW');
        assert(publications().length === 0 && intervals.size === 0, 'idle traffic');
    });
    const axes = {
        KeyW: [0.2, 0, 0], KeyS: [-0.2, 0, 0], KeyA: [0, 0.2, 0],
        KeyD: [0, -0.2, 0], KeyJ: [0, 0, 0.5], KeyK: [0, 0, -0.5]
    };
    for (const [code, values] of Object.entries(axes)) {
        check(code + ' direction and release', () => {
            enable();
            key(code);
            const msg = last();
            assert([msg.linear.x, msg.linear.y, msg.angular.z].every((v, i) => v === values[i]), 'wrong direction');
            assert(document.querySelector(`[data-drive-key="${code}"]`).classList.contains('pressed'), 'key highlight');
            key(code, 'keyup'); zero();
            assert(intervals.size === 0, 'timer stopped on release');
        });
    }
    check('held keys publish repeatedly without autorepeat timers', () => {
        key('KeyW');
        const before = publications().length;
        key('KeyW', 'keydown', {repeat: true});
        assert(intervals.size === 1 && publications().length === before, 'repeat duplication');
        intervals.forEach(callback => callback());
        assert(publications().length === before + 1 && last().linear.x === 0.2, 'heartbeat');
        key('KeyW', 'keyup');
    });
    check('diagonal normalization and partial release', () => {
        key('KeyW'); key('KeyA'); key('KeyJ');
        const msg = last();
        assert(Math.abs(Math.hypot(msg.linear.x, msg.linear.y) - 0.2) < 1e-9 && msg.angular.z === 0.5, 'diagonal');
        key('KeyW', 'keyup');
        assert(last().linear.x === 0 && last().linear.y === 0.2 && last().angular.z === 0.5, 'partial release');
        key('KeyA', 'keyup'); key('KeyJ', 'keyup'); zero();
    });
    check('opposite directions cancel', () => {
        Object.keys(axes).forEach(code => key(code)); zero();
        Object.keys(axes).forEach(code => key(code, 'keyup')); zero();
    });
    check('space stops and cannot resume via autorepeat', () => {
        key('KeyW'); key('Space'); zero();
        assert(!active() && intervals.size === 0, 'must disable');
        enable(); key('KeyW', 'keydown', {repeat: true});
        zero(); assert(intervals.size === 0, 'held key resumed');
    });
    check('Escape stops', () => { key('KeyW'); key('Escape'); zero(); assert(!active(), 'enabled'); });
    check('stop button works', () => { enable(); key('KeyW'); stop.click(); zero(); assert(!active(), 'enabled'); });
    check('toggle off stops', () => { enable(); key('KeyW'); toggle.click(); zero(); assert(!active(), 'enabled'); });
    check('window blur stops', () => { enable(); key('KeyW'); window.dispatchEvent(new Event('blur')); zero(); assert(!active(), 'enabled'); });
    check('hidden page stops', () => {
        enable(); key('KeyW');
        Object.defineProperty(document, 'hidden', {value: true, configurable: true});
        document.dispatchEvent(new Event('visibilitychange'));
        zero(); assert(!active(), 'enabled');
        delete document.hidden;
    });
    check('editable focus stops and typing is ignored', () => {
        const input = document.createElement('input'); document.body.append(input);
        enable(); key('KeyW'); input.focus(); zero(); assert(!active(), 'enabled');
        const before = publications().length; key('KeyW', 'keydown', {}, input);
        assert(publications().length === before, 'typing moved robot'); input.remove();
    });
    check('contenteditable and IME are ignored', () => {
        const input = document.createElement('div'); input.contentEditable = 'true'; document.body.append(input);
        enable(); key('KeyW', 'keydown', {}, input); zero(); assert(!active(), 'editable enabled');
        input.remove(); enable(); key('KeyW', 'keydown', {isComposing: true}); zero(); assert(!active(), 'IME enabled');
    });
    check('modifier shortcuts stop driving', () => {
        enable(); key('KeyW'); key('KeyS', 'keydown', {ctrlKey: true}); zero(); assert(!active(), 'enabled');
    });
    check('physical uppercase keys work', () => {
        enable(); key('KeyW', 'keydown', {key: 'W', shiftKey: true});
        assert(last().linear.x === 0.2, 'uppercase'); key('KeyW', 'keyup');
    });
    check('speed sliders update velocity and label', () => {
        for (const [id, value] of [['teleop-linear', '0.35'], ['teleop-angular', '0.80']]) {
            const slider = document.getElementById(id); slider.value = value;
            slider.dispatchEvent(new Event('input'));
            assert(document.getElementById(id + '-value').textContent.includes(value), 'label');
        }
        key('KeyW'); key('KeyK');
        assert(last().linear.x === 0.35 && last().angular.z === -0.8, 'speed');
        stop.click();
    });
    check('disconnect clears motion and reconnect stays disabled', () => {
        enable(); key('KeyW'); current().close();
        assert(!active() && toggle.disabled && intervals.size === 0, 'disconnect state');
        assert(retries.size === 1, 'reconnect scheduled');
        const callback = [...retries.values()][0]; retries.clear(); callback(); current().open();
        assert(!active() && publications().length === 0, 'replayed motion');
        enable(); key('KeyW', 'keydown', {repeat: true});
        assert(publications().length === 0, 'replayed held key');
    });
    check('congested socket closes instead of queueing motion', () => {
        current().bufferedAmount = 10; key('KeyW');
        assert(!active() && intervals.size === 0 && current().readyState === 3, 'congestion state');
        assert(publications().length === 0, 'queued motion');
        const callback = [...retries.values()][0]; retries.clear(); callback(); current().open();
    });
    check('rosbridge errors disable driving', () => {
        enable(); key('KeyW');
        current().dispatchEvent(new MessageEvent('message', {data: JSON.stringify({op: 'status', level: 'error'})}));
        zero(); assert(!active(), 'enabled');
        const callback = [...retries.values()][0]; retries.clear(); callback(); current().open();
    });
    check('page exit stops and cancels reconnect', () => {
        enable(); key('KeyW'); window.dispatchEvent(new Event('pagehide'));
        zero(); assert(!active() && retries.size === 0 && intervals.size === 0, 'exit state');
    });
    check('back-forward restoration reconnects without driving', () => {
        window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted: true})); current().open();
        assert(!active() && !toggle.disabled && publications().length === 0, 'restore');
    });
    const result = document.createElement('pre'); result.id = 'test-results';
    result.hidden = true;
    result.textContent = JSON.stringify({passed, failures}); document.body.append(result);
    window.scrollTo(0, 0);
});
