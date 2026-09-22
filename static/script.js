// Reference upload, cat/person test mode and visual recognition lifecycle.
let uploadedFilePath = null, selectedFile = null, statusCheckInterval = null;
let recognition = null, revision = 0, switching = false, starting = false, uploading = false, polling = false;
const byId = id => document.getElementById(id);
const fileInput = byId('file-input'), uploadArea = byId('upload-area');
const uploadPlaceholder = byId('upload-placeholder'), previewContainer = byId('preview-container');
const previewImage = byId('preview-image'), uploadBtn = byId('upload-btn');
const startBtn = byId('start-btn'), stopBtn = byId('stop-btn'), modeBtn = byId('test-mode-toggle');
const videoOverlay = byId('video-overlay'), videoStream = byId('video-stream');
const startCameraBtn = byId('start-camera-btn');

function renderRecognition() {
    const person = recognition?.mode === 'person';
    const busy = Boolean(starting || recognition?.tracking_loading || recognition?.tracking_active);
    modeBtn.disabled = switching || !recognition;
    modeBtn.textContent = switching ? '正在切换…' : person ? '退出测试模式（识别猫）' : '切换到人物测试模式';
    modeBtn.setAttribute('aria-pressed', String(person));
    byId('tracking-mode-status').textContent = '当前模式：' + (person ? '人物测试 · ReID 重识别' : '猫咪识别');
    byId('reference-hint').textContent = person
        ? '上传仅含一人的清晰照片，尽量完整显示当前衣着。连续确认后标注；衣着相似或遮挡时可能无法区分，请重新取景。'
        : '上传目标猫的清晰照片。';
    byId('upload-prompt').textContent = person ? '点击选择或拖拽目标人物照片' : '点击选择或拖拽猫照片';
    uploadBtn.disabled = !recognition || switching || uploading || busy || !selectedFile;
    startBtn.disabled = !recognition || switching || uploading || busy || !uploadedFilePath;
    startBtn.textContent = starting || recognition?.tracking_loading ? '正在初始化…' : '🚀 启动追踪';
    stopBtn.disabled = switching || !busy;
    byId('status-dot').className = 'status-dot ' + (busy ? 'tracking' : 'ready');
    byId('status-text').textContent = starting || recognition?.tracking_loading ? '模型初始化中'
        : recognition?.tracking_active ? (person ? '人物识别中' : '猫咪识别中') : recognition?.tracking_error || '就绪';
}

function resetUpload() {
    selectedFile = null; uploadedFilePath = null;
    fileInput.value = ''; previewImage.removeAttribute('src');
    uploadPlaceholder.style.display = 'block'; previewContainer.style.display = 'none';
    renderRecognition();
}

function applyRecognition(state) {
    if (recognition && (state.mode !== recognition.mode || state.generation !== recognition.generation)) {
        revision++; starting = false; uploading = false;
        resetUpload();
    }
    recognition = state;
    renderRecognition();
}

async function jsonRequest(url, options = {}) {
    const response = await fetch(url, {cache: 'no-store', ...options});
    const data = await response.json();
    if (!response.ok || data.success === false) throw new Error(data.message || '请求失败');
    return data;
}

function handleFile(file) {
    if (switching || uploading || starting || recognition?.tracking_active || recognition?.tracking_loading) return;
    if (!['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'].includes(file.type)) {
        showToast('请上传有效的图片文件', 'error'); return;
    }
    if (file.size > 16 * 1024 * 1024) { showToast('文件不能超过 16 MB', 'error'); return; }
    selectedFile = file; uploadedFilePath = null;
    const version = ++revision;
    const reader = new FileReader();
    reader.onload = event => {
        if (version !== revision) return;
        previewImage.src = event.target.result;
        uploadPlaceholder.style.display = 'none'; previewContainer.style.display = 'block';
        renderRecognition();
    };
    reader.readAsDataURL(file);
    renderRecognition();
}

async function uploadFile() {
    if (!recognition || !selectedFile || uploading || switching) return;
    const version = ++revision;
    const form = new FormData();
    form.append('file', selectedFile); form.append('mode', recognition.mode);
    form.append('generation', recognition.generation);
    uploading = true; renderRecognition();
    try {
        const result = await jsonRequest('/upload', {method: 'POST', body: form});
        if (version !== revision) return;
        uploadedFilePath = result.filepath;
        showToast(result.message);
    } catch (error) { if (version === revision) showToast(error.message, 'error'); }
    finally { if (version === revision) { uploading = false; renderRecognition(); } }
}

async function startTracking() {
    if (!uploadedFilePath || !recognition || switching || starting) return;
    const version = ++revision;
    starting = true; renderRecognition();
    try {
        const result = await jsonRequest('/start_tracking', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filepath: uploadedFilePath, mode: recognition.mode, generation: recognition.generation})});
        if (version !== revision) return;
        recognition.tracking_active = true; recognition.tracking_loading = false;
        showToast(result.message);
    } catch (error) { if (version === revision) showToast(error.message, 'error'); }
    finally { if (version === revision) { starting = false; renderRecognition(); } }
}

async function stopTracking() {
    const version = ++revision;
    starting = false;
    try {
        await jsonRequest('/stop_tracking', {method: 'POST'});
        if (version !== revision) return;
        recognition.tracking_active = false; recognition.tracking_loading = false;
        showToast('识别与自动追踪已停止');
    } catch (error) { if (version === revision) showToast(error.message, 'error'); }
    finally { if (version === revision) renderRecognition(); }
}

async function switchMode() {
    if (switching || !recognition) return;
    const version = ++revision;
    switching = true; starting = false; uploading = false; renderRecognition();
    try {
        const result = await jsonRequest('/tracking_mode', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({mode: recognition.mode === 'cat' ? 'person' : 'cat'})});
        if (version !== revision) return;
        recognition = result; resetUpload();
        showToast('模式已切换，请上传对应目标的新参考照片');
    } catch (error) { if (version === revision) showToast(error.message, 'error'); }
    finally { switching = false; renderRecognition(); }
}

async function pollStatus() {
    if (polling || switching) return;
    polling = true;
    const version = revision;
    try {
        const data = await jsonRequest('/status');
        if (version !== revision) return;
        applyRecognition(data);
        const location = data.cat_location;
        if (location) byId('cat-location-status').textContent = '地图标注：' + location.message
            + (location.position ? ` · ${location.position.map(v => v.toFixed(2)).join(', ')} m` : '');
        videoOverlay.classList.toggle('hidden', data.camera_active);
        if (!data.camera_active) { startCameraBtn.disabled = false; startCameraBtn.textContent = '启动相机'; }
    } catch (error) { console.error('状态检查失败:', error); }
    finally { polling = false; }
}

async function startCamera() {
    startCameraBtn.disabled = true;
    try {
        await jsonRequest('/start_camera', {method: 'POST'});
        videoOverlay.classList.add('hidden'); videoStream.src = '/video_feed?' + Date.now();
    } catch (error) { showToast(error.message, 'error'); startCameraBtn.disabled = false; }
}

function showToast(message, type = 'success') {
    const toast = byId('toast'); toast.textContent = message; toast.className = 'toast ' + type + ' show';
    setTimeout(() => toast.classList.remove('show'), 3000);
}

document.addEventListener('DOMContentLoaded', () => {
    uploadArea.addEventListener('click', () => { if (!selectedFile) fileInput.click(); });
    fileInput.addEventListener('change', event => { if (event.target.files[0]) handleFile(event.target.files[0]); });
    uploadArea.addEventListener('dragover', event => { event.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', event => {
        event.preventDefault(); uploadArea.classList.remove('dragover');
        if (event.dataTransfer.files[0]) handleFile(event.dataTransfer.files[0]);
    });
    byId('remove-image').addEventListener('click', event => {
        event.stopPropagation(); revision++; uploading = false; resetUpload();
    });
    uploadBtn.addEventListener('click', uploadFile); startBtn.addEventListener('click', startTracking);
    stopBtn.addEventListener('click', stopTracking); modeBtn.addEventListener('click', switchMode);
    startCameraBtn.addEventListener('click', startCamera);
    renderRecognition(); pollStatus(); statusCheckInterval = setInterval(pollStatus, 1000);
});
window.addEventListener('beforeunload', () => clearInterval(statusCheckInterval));
