// 全局变量
let uploadedFilePath = null;
let statusCheckInterval = null;

// DOM元素
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const removeImageBtn = document.getElementById('remove-image');
const uploadBtn = document.getElementById('upload-btn');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const toast = document.getElementById('toast');
const startCameraBtn = document.getElementById('start-camera-btn');
const videoOverlay = document.getElementById('video-overlay');
const videoStream = document.getElementById('video-stream');

// 页面加载完成
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    startStatusPolling();
});

// 初始化事件监听器
function initializeEventListeners() {
    // 点击上传区域
    uploadArea.addEventListener('click', () => {
        if (!previewContainer.style.display || previewContainer.style.display === 'none') {
            fileInput.click();
        }
    });

    // 文件选择
    fileInput.addEventListener('change', handleFileSelect);

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 移除图片
    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // 上传按钮
    uploadBtn.addEventListener('click', uploadFile);

    // 启动追踪
    startBtn.addEventListener('click', startTracking);

    // 停止追踪
    stopBtn.addEventListener('click', stopTracking);
    
    // 启动相机
    startCameraBtn.addEventListener('click', startCamera);
}

// 处理文件选择
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// 处理文件
function handleFile(file) {
    // 验证文件类型
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showToast('请上传有效的图片文件！', 'error');
        return;
    }

    // 验证文件大小（16MB）
    if (file.size > 16 * 1024 * 1024) {
        showToast('文件大小不能超过16MB！', 'error');
        return;
    }

    // 显示预览
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        previewContainer.style.display = 'block';
        uploadBtn.disabled = false;
        
        // 重置上传状态
        uploadedFilePath = null;
        startBtn.disabled = true;
    };
    reader.readAsDataURL(file);
}

// 重置上传
function resetUpload() {
    fileInput.value = '';
    uploadPlaceholder.style.display = 'block';
    previewContainer.style.display = 'none';
    uploadBtn.disabled = true;
    startBtn.disabled = true;
    uploadedFilePath = null;
}

// 上传文件
async function uploadFile() {
    const file = fileInput.files[0];
    if (!file) {
        showToast('请先选择文件！', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    uploadBtn.disabled = true;
    uploadBtn.textContent = '上传中...';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            uploadedFilePath = data.filepath;
            showToast(data.message, 'success');
            startBtn.disabled = false;
            uploadBtn.textContent = '重新上传';
            uploadBtn.disabled = false;
        } else {
            showToast(data.message, 'error');
            uploadBtn.textContent = '上传照片';
            uploadBtn.disabled = false;
        }
    } catch (error) {
        showToast('上传失败：' + error.message, 'error');
        uploadBtn.textContent = '上传照片';
        uploadBtn.disabled = false;
    }
}

// 启动追踪
async function startTracking() {
    if (!uploadedFilePath) {
        showToast('请先上传参考照片！', 'error');
        return;
    }

    startBtn.disabled = true;
    startBtn.textContent = '正在启动...';

    try {
        const response = await fetch('/start_tracking', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: uploadedFilePath
            })
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.message, 'success');
            stopBtn.disabled = false;
            updateStatus('tracking', '追踪中');
        } else {
            showToast(data.message, 'error');
            startBtn.disabled = false;
            startBtn.textContent = '🚀 启动追踪';
        }
    } catch (error) {
        showToast('启动失败：' + error.message, 'error');
        startBtn.disabled = false;
        startBtn.textContent = '🚀 启动追踪';
    }
}

// 停止追踪
async function stopTracking() {
    stopBtn.disabled = true;
    stopBtn.textContent = '正在停止...';

    try {
        const response = await fetch('/stop_tracking', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.message, 'success');
            updateStatus('ready', '就绪');
            startBtn.disabled = false;
            startBtn.textContent = '🚀 启动追踪';
            stopBtn.textContent = '⏹ 停止追踪';
        } else {
            showToast(data.message, 'error');
            stopBtn.disabled = false;
            stopBtn.textContent = '⏹ 停止追踪';
        }
    } catch (error) {
        showToast('停止失败：' + error.message, 'error');
        stopBtn.disabled = false;
        stopBtn.textContent = '⏹ 停止追踪';
    }
}

// 启动相机
async function startCamera() {
    startCameraBtn.disabled = true;
    startCameraBtn.textContent = '启动中...';

    try {
        const response = await fetch('/start_camera', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.message, 'success');
            videoOverlay.classList.add('hidden');
            // 刷新视频流
            videoStream.src = '/video_feed?' + new Date().getTime();
        } else {
            showToast(data.message, 'error');
            startCameraBtn.disabled = false;
            startCameraBtn.textContent = '启动相机';
        }
    } catch (error) {
        showToast('启动相机失败：' + error.message, 'error');
        startCameraBtn.disabled = false;
        startCameraBtn.textContent = '启动相机';
    }
}

// 轮询状态
function startStatusPolling() {
    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            // 更新相机状态
            if (data.camera_active) {
                videoOverlay.classList.add('hidden');
            } else {
                videoOverlay.classList.remove('hidden');
                startCameraBtn.disabled = false;
                startCameraBtn.textContent = '启动相机';
            }

            // 更新追踪状态
            if (data.tracking_active) {
                updateStatus('tracking', '追踪中');
                stopBtn.disabled = false;
                startBtn.disabled = true;
            } else {
                updateStatus('ready', '就绪');
                stopBtn.disabled = true;
                if (uploadedFilePath && data.camera_active) {
                    startBtn.disabled = false;
                    startBtn.textContent = '🚀 启动追踪';
                }
            }
        } catch (error) {
            console.error('状态检查失败:', error);
        }
    }, 2000); // 每2秒检查一次
}

// 更新状态显示
function updateStatus(status, text) {
    statusDot.className = 'status-dot ' + status;
    statusText.textContent = text;
}

// 显示提示消息
function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.className = 'toast ' + type + ' show';

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// 页面卸载时停止轮询
window.addEventListener('beforeunload', () => {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
});
