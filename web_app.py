"""
Cat Tracking Web Application
用于上传参考猫照片并启动追踪的Web界面
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import threading
import time
from werkzeug.utils import secure_filename
from track_specific_cat import CatTracker
import cv2
import numpy as np
import pyrealsense2 as rs

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量存储追踪器状态
tracker_instance = None
tracking_thread = None
tracking_active = False

# RealSense相机相关
pipeline = None
align = None
frame_lock = threading.Lock()
current_frame = None
current_depth_frame = None
camera_active = False


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_camera():
    """初始化RealSense相机"""
    global pipeline, align, camera_active
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)
        camera_active = True
        print("✓ RealSense相机初始化成功")
        return True
    except Exception as e:
        print(f"✗ 相机初始化失败: {e}")
        camera_active = False
        return False


def stop_camera():
    """停止RealSense相机"""
    global pipeline, camera_active
    if pipeline:
        try:
            pipeline.stop()
            camera_active = False
            print("✓ 相机已停止")
        except:
            pass


def generate_frames():
    """生成视频流帧"""
    global pipeline, align, current_frame, current_depth_frame, tracking_active, tracker_instance, frame_lock
    
    if not camera_active:
        if not init_camera():
            return
    
    frame_count = 0
    
    while camera_active:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            with frame_lock:
                current_frame = color_image.copy()
                current_depth_frame = depth_frame
            
            frame_count += 1
            display_image = color_image.copy()
            
            # 如果正在追踪，进行检测和绘制
            if tracking_active and tracker_instance and frame_count % 2 == 0:
                try:
                    results = tracker_instance.yolo_model(color_image, verbose=False)
                    cats = []
                    
                    if results[0].boxes is not None:
                        for detection in results[0].boxes:
                            class_id = int(detection.cls)
                            confidence = float(detection.conf)
                            
                            if class_id == 15 and confidence > 0.5:
                                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                                w = x2 - x1
                                h = y2 - y1
                                x = x1
                                y = y1
                                
                                x = max(0, x)
                                y = max(0, y)
                                w = min(w, color_image.shape[1] - x)
                                h = min(h, color_image.shape[0] - y)
                                
                                if w > 30 and h > 30:
                                    cat_roi = color_image[y:y+h, x:x+w]
                                    match_score = tracker_instance.match_cat(cat_roi)
                                    cats.append({
                                        'box': (x, y, w, h),
                                        'confidence': confidence,
                                        'match_score': match_score
                                    })
                    
                    # 找到最匹配的猫
                    target_cat = None
                    if cats:
                        sorted_cats = sorted(cats, key=lambda c: c['match_score'], reverse=True)
                        target_cat = sorted_cats[0]
                        
                        if target_cat['match_score'] < 0.75:
                            target_cat = None
                        elif len(cats) > 1:
                            score_gap = target_cat['match_score'] - sorted_cats[1]['match_score']
                            if score_gap < 0.05:
                                target_cat = None
                    
                    # 绘制结果
                    for cat in cats:
                        x, y, w, h = cat['box']
                        match_score = cat['match_score']
                        
                        if target_cat and cat == target_cat:
                            color = (0, 255, 0)
                            thickness = 3
                            center_x = x + w // 2
                            center_y = y + h // 2
                            distance = depth_frame.get_distance(center_x, center_y)
                            
                            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, thickness)
                            label = f"TARGET {distance:.2f}m"
                            label2 = f"Score: {match_score:.3f}"
                            
                            cv2.rectangle(display_image, (x, y - 50), (x + 300, y), color, -1)
                            cv2.putText(display_image, label, (x + 5, y - 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            cv2.putText(display_image, label2, (x + 5, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            cv2.drawMarker(display_image, (center_x, center_y), 
                                         color, cv2.MARKER_CROSS, 20, 2)
                        else:
                            color = (128, 128, 128)
                            thickness = 1
                            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, thickness)
                            cv2.putText(display_image, f"Other ({match_score:.3f})", 
                                      (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # 显示追踪状态
                    status_text = "TRACKING" if target_cat else "NO TARGET"
                    status_color = (0, 255, 0) if target_cat else (0, 0, 255)
                    cv2.putText(display_image, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    
                except Exception as e:
                    print(f"追踪处理错误: {e}")
            
            # 添加帧信息
            cv2.putText(display_image, f"FPS: {30 if camera_active else 0}", 
                       (display_image.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', display_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30fps
            
        except Exception as e:
            print(f"帧生成错误: {e}")
            time.sleep(0.1)


def run_tracker(image_path):
    """初始化追踪器（不运行run方法）"""
    global tracking_active, tracker_instance
    try:
        tracker_instance = CatTracker(image_path)
        tracking_active = True
        print("✓ 追踪器初始化成功")
    except Exception as e:
        print(f"✗ 追踪器初始化错误: {e}")
        tracking_active = False


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    global tracking_active, tracker_instance, tracking_thread
    
    # 检查是否已经在追踪
    if tracking_active:
        return jsonify({
            'success': False,
            'message': '追踪正在运行中，请先停止当前追踪'
        }), 400
    
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': '没有上传文件'
        }), 400
    
    file = request.files['file']
    
    # 检查文件名
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': '未选择文件'
        }), 400
    
    # 验证并保存文件
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 验证图像是否可读取
        test_img = cv2.imread(filepath)
        if test_img is None:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'message': '无法读取图像文件，请确保上传的是有效的图片'
            }), 400
        
        return jsonify({
            'success': True,
            'message': '文件上传成功！',
            'filename': filename,
            'filepath': filepath
        })
    else:
        return jsonify({
            'success': False,
            'message': f'不支持的文件格式，请上传: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400


@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    """启动追踪"""
    global tracking_active, tracker_instance
    
    if tracking_active:
        return jsonify({
            'success': False,
            'message': '追踪已经在运行中'
        }), 400
    
    data = request.get_json()
    filepath = data.get('filepath')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'message': '参考图片文件不存在'
        }), 400
    
    # 初始化相机（如果还没有初始化）
    if not camera_active:
        if not init_camera():
            return jsonify({
                'success': False,
                'message': '相机初始化失败，请检查RealSense相机是否连接'
            }), 500
    
    # 初始化追踪器
    run_tracker(filepath)
    
    if tracking_active:
        return jsonify({
            'success': True,
            'message': '追踪已启动！请查看浏览器中的视频流。'
        })
    else:
        return jsonify({
            'success': False,
            'message': '追踪器初始化失败'
        }), 500


@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    """停止追踪"""
    global tracking_active, tracker_instance
    
    if not tracking_active:
        return jsonify({
            'success': False,
            'message': '当前没有运行中的追踪'
        }), 400
    
    tracking_active = False
    tracker_instance = None
    
    return jsonify({
        'success': True,
        'message': '追踪已停止'
    })


@app.route('/status', methods=['GET'])
def get_status():
    """获取追踪状态"""
    return jsonify({
        'tracking_active': tracking_active,
        'has_tracker': tracker_instance is not None,
        'camera_active': camera_active
    })


@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    """启动相机"""
    if camera_active:
        return jsonify({'success': True, 'message': '相机已经在运行中'})
    
    if init_camera():
        return jsonify({'success': True, 'message': '相机启动成功'})
    else:
        return jsonify({'success': False, 'message': '相机启动失败，请检查RealSense连接'}), 500


@app.route('/stop_camera', methods=['POST'])
def stop_camera_route():
    """停止相机"""
    global tracking_active
    if tracking_active:
        return jsonify({'success': False, 'message': '请先停止追踪'}), 400
    
    stop_camera()
    return jsonify({'success': True, 'message': '相机已停止'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("="*60)
    print("Cat Tracking Web Application")
    print("="*60)
    print("\n正在启动Web服务器...")
    print("请在浏览器中访问: http://localhost:5000")
    print("\n使用说明:")
    print("1. 在浏览器中查看实时视频流")
    print("2. 上传一张参考猫的照片")
    print("3. 点击'启动追踪'按钮")
    print("4. 在视频流中查看实时追踪结果")
    print("\n按 Ctrl+C 停止Web服务器")
    print("="*60)
    print()
    
    # 启动时自动初始化相机
    print("→ 正在初始化RealSense相机...")
    if init_camera():
        print("✓ 相机初始化成功\n")
    else:
        print("✗ 相机初始化失败，请检查RealSense相机连接")
        print("服务器仍会启动，但需要手动启动相机\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    finally:
        print("\n正在关闭相机...")
        stop_camera()
        print("服务器已停止")
