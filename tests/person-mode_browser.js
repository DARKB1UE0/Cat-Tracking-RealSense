const requests=[], intervals=[];
window.setInterval=fn=>{intervals.push(fn);return intervals.length;};window.clearInterval=()=>{};
window.FileReader=class{readAsDataURL(){this.onload({target:{result:'data:image/png;base64,iVBORw0KGgo='}});}};
let state={mode:'cat',generation:0,tracking_active:false,tracking_loading:false,camera_active:true};
let deferStart=false,finishStart,failSwitch=false;
window.fetch=async(url,options)=>{
    const reply=(data,ok=true)=>({ok,json:async()=>data});
    const data=options.body instanceof FormData ? Object.fromEntries(options.body.entries())
        : options.body ? JSON.parse(options.body) : {};
    requests.push({url,data});
    if(url==='/status')return reply({...state});
    if(url==='/tracking_mode'){
        if(failSwitch)return reply({success:false,message:'切换失败'},false);
        state={...state,mode:data.mode,generation:state.generation+1,tracking_active:false,tracking_loading:false};
        return reply({success:true,...state});
    }
    if(url==='/upload')return reply({success:true,filepath:'/uploads/ref-'+state.generation+'.jpg',message:'已上传'});
    if(url==='/start_tracking'){
        if(deferStart)return new Promise(resolve=>{finishStart=()=>resolve(reply({success:true,message:'旧初始化完成'}));});
        state.tracking_active=true;return reply({success:true,message:'已启动'});
    }
    if(url==='/stop_tracking'){state.tracking_active=false;state.tracking_loading=false;return reply({success:true});}
    throw new Error('unexpected request '+url);
};
window.addEventListener('load',async()=>{
    const passed=[],failures=[];
    const settle=async()=>{for(let i=0;i<50;i++)await Promise.resolve();};
    const tick=async()=>{intervals.forEach(fn=>fn());await settle();};
    const click=async id=>{document.getElementById(id).click();await settle();};
    const assert=(ok,msg)=>{if(!ok)throw new Error(msg);};
    const check=async(name,fn)=>{try{await fn();passed.push(name);}catch(e){failures.push(name+': '+e.message);}};
    const select=()=>{
        const input=document.getElementById('file-input');
        Object.defineProperty(input,'files',{value:[new File(['image'],'reference.jpg',{type:'image/jpeg'})],configurable:true});
        input.dispatchEvent(new Event('change'));
    };
    await settle();
    await check('default cat mode never starts recognition or motion',async()=>{
        assert(document.getElementById('test-mode-toggle').getAttribute('aria-pressed')==='false','not cat');
        assert(requests.every(r=>r.url==='/status'),'unsolicited start');
    });
    await check('test button changes labels and requires a new reference',async()=>{
        select();await click('upload-btn');assert(!document.getElementById('start-btn').disabled,'cat reference missing');
        await click('test-mode-toggle');
        assert(state.mode==='person' && document.getElementById('start-btn').disabled,'old reference retained');
        assert(document.getElementById('reference-hint').textContent.includes('一人'),'person instructions');
        assert(document.getElementById('upload-prompt').textContent.includes('人物'),'upload instructions');
    });
    await check('person reference upload and recognition start carry mode generation',async()=>{
        select();await click('upload-btn');await click('start-btn');
        const uploaded=requests.findLast(r=>r.url==='/upload').data;
        const started=requests.findLast(r=>r.url==='/start_tracking').data;
        assert(uploaded.mode==='person' && uploaded.generation==='1','upload mode');
        assert(started.mode==='person' && started.generation===1 && state.tracking_active,'start mode');
        assert(!requests.some(r=>r.url==='/api/follow/control'),'started driving automatically');
    });
    await check('exit test mode clears active target and uploaded photo',async()=>{
        await click('test-mode-toggle');
        assert(state.mode==='cat' && !state.tracking_active,'did not leave person mode');
        assert(document.getElementById('start-btn').disabled && document.getElementById('preview-container').style.display==='none','stale reference');
    });
    await check('late model response cannot reactivate mode after switch',async()=>{
        select();await click('upload-btn');deferStart=true;await click('start-btn');
        await click('test-mode-toggle');finishStart();await settle();deferStart=false;
        assert(state.mode==='person' && !state.tracking_active,'old response resumed');
        assert(document.getElementById('start-btn').disabled,'old reference enabled');
    });
    await check('other page mode change clears local reference and failed switch preserves mode',async()=>{
        select();await click('upload-btn');state={...state,mode:'cat',generation:state.generation+1};await tick();
        assert(document.getElementById('start-btn').disabled,'external mode kept reference');
        failSwitch=true;await click('test-mode-toggle');failSwitch=false;
        assert(document.getElementById('test-mode-toggle').getAttribute('aria-pressed')==='false','failed switch changed mode');
    });
    const report=document.createElement('pre');report.id='test-results';report.hidden=true;
    report.textContent=JSON.stringify({passed,failures});document.body.append(report);
});
