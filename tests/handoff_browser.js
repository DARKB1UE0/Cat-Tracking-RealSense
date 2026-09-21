// Load teleop, gimbal and following together with fake HTTP and rosbridge.
const intervals = new Map(); let nextId = 0;
window.setInterval = (callback, period) => { intervals.set(++nextId,{callback,period}); return nextId; };
window.clearInterval = id => intervals.delete(id);
const sockets=[], requests=[];
window.WebSocket = class extends EventTarget {
    static OPEN=1;
    constructor() { super(); this.readyState=0; this.bufferedAmount=0; this.sent=[]; sockets.push(this); }
    send(data) { this.sent.push(JSON.parse(data)); }
    close() { this.readyState=3; this.dispatchEvent(new Event('close')); }
};
let following=false, gimbalOwner=null, pendingTakeover;
window.fetch = async (url, options) => {
    const reply=data=>({ok:true,json:async()=>data});
    const data=options.body ? JSON.parse(options.body) : {};
    requests.push({url,...data});
    if (url==='/api/teleop/enable') {
        following=false; gimbalOwner=null;
        return new Promise(resolve=>{ pendingTakeover=()=>resolve(reply({ok:true})); });
    }
    if (url==='/api/follow/status') return reply({ready:true,active:following,stopping:false,message:'状态'});
    if (url==='/api/gimbal/status') return reply({connected:true,controllable:true,active:!!gimbalOwner,
        owner:gimbalOwner,yaw:0,pitch:0,fault:0,enabled:true});
    if (url==='/api/follow/control') {
        if (data.action==='start') { following=true;gimbalOwner='auto'; }
        if (data.action==='stop') { following=false;gimbalOwner=null; }
        return reply({ok:true,token:'follow-token'});
    }
    if (data.action==='start') gimbalOwner='manual';
    if (data.action==='stop') gimbalOwner=null;
    return reply({ok:true,token:'gimbal-token',yaw:0,pitch:0});
};
window.addEventListener('load',async()=>{
    const passed=[],failures=[];
    const settle=async()=>{for(let i=0;i<60;i++)await Promise.resolve();};
    const tick=async period=>{intervals.forEach(t=>{if(t.period===period)t.callback();});await settle();};
    const click=async id=>{document.getElementById(id).click();await settle();};
    const active=id=>document.getElementById(id).getAttribute('aria-pressed')==='true';
    const assert=(ok,msg)=>{if(!ok)throw new Error(msg);};
    const check=async(name,fn)=>{try{await fn();passed.push(name);}catch(e){failures.push(name+': '+e.message);}};
    const socket=sockets[0];socket.readyState=1;socket.dispatchEvent(new Event('open'));await settle();
    await check('manual gimbal releases before following acquires it',async()=>{
        await click('gimbal-toggle'); assert(gimbalOwner==='manual','manual gimbal');
        await click('follow-start'); assert(following && gimbalOwner==='auto','follow handoff');
        const started=requests.findIndex(r=>r.url==='/api/follow/control'&&r.action==='start');
        assert(requests.slice(0,started).some(r=>r.url==='/api/gimbal/control'&&r.action==='stop'),'release missing');
    });
    await check('keyboard can take over active following while status polling continues',async()=>{
        await tick(300); await tick(250);
        assert(!document.getElementById('teleop-toggle').disabled,'takeover locked');
        await click('teleop-toggle');
        assert(!following && !active('teleop-toggle'),'must cancel before enable');
        await tick(300); await tick(250);pendingTakeover();await settle();
        assert(active('teleop-toggle') && !active('follow-start'),'keyboard takeover');
        document.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyW',bubbles:true}));
        assert(socket.sent.at(-1).msg.linear.x===.2,'manual velocity');
        document.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyW',bubbles:true}));
    });
    await check('following disables keyboard and clears held keys',async()=>{
        await click('follow-start');await tick(300);
        assert(following && !active('teleop-toggle'),'keyboard remained active');
        const count=socket.sent.length;
        document.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyW',bubbles:true}));
        assert(socket.sent.length===count,'manual motion during following');
    });
    await check('blur during combined takeover does not re-enable on response',async()=>{
        await click('teleop-toggle');window.dispatchEvent(new Event('blur'));await settle();
        pendingTakeover();await settle();await tick(300);
        assert(!active('teleop-toggle') && !following,'late takeover enabled');
    });
    const report=document.createElement('pre');report.id='test-results';report.hidden=true;
    report.textContent=JSON.stringify({passed,failures});document.body.append(report);
});
