import pyrealsense2 as rs
import cv2
import numpy as np
import sys
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from ultralytics import YOLO
from scipy.spatial.distance import cosine

class CatTracker:
    def __init__(self, reference_image_path):
        """
        初始化猫追踪器（YOLOv8 + ResNet方案）
        
        参数:
            reference_image_path: 参考猫图片的路径
        """
        # 加载参考图片
        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise ValueError(f"无法读取参考图片: {reference_image_path}")
        
        print(f"已加载参考图片: {reference_image_path}")
        
        # 1. 加载YOLOv8模型
        print("正在加载YOLOv8模型...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # nano版本，速度快
            print("YOLOv8加载成功")
        except:
            print("YOLOv8模型不存在，正在下载...")
            self.yolo_model = YOLO('yolov8n.pt')
        
        # 2. 加载ResNet50特征提取器
        print("正在加载ResNet50特征提取器...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载预训练ResNet50
        self.resnet = resnet50(pretrained=True)
        self.resnet = self.resnet.to(self.device)
        
        # 移除最后的分类层，保留特征层
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()  # 设置为评估模式
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 3. 从参考图片中提取目标猫的特征
        print("正在从参考图片中提取目标猫的特征...")
        self.reference_feature = self._extract_feature_from_image(self.reference_image)
        print(f"参考特征维度: {self.reference_feature.shape}")
        print("初始化完成！")
    
    def _extract_feature_from_image(self, image):
        """
        从图像中提取ResNet特征
        """
        try:
            # 转换图像格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PIL转换
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image_rgb)
            
            # 预处理
            tensor = self.preprocess(pil_image)
            tensor = tensor.unsqueeze(0).to(self.device)  # 添加batch维度
            
            # 提取特征
            with torch.no_grad():
                feature = self.resnet(tensor)
            
            # 归一化
            feature = feature.squeeze().cpu().numpy()
            feature = feature / (np.linalg.norm(feature) + 1e-6)
            
            return feature
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def match_cat(self, cat_image):
        """
        使用余弦相似度比较ResNet特征
        
        返回:
            相似度分数 (0-1之间，越高越相似)
        """
        try:
            feature = self._extract_feature_from_image(cat_image)
            
            if feature is None:
                return 0
            
            # 计算余弦相似度
            similarity = 1.0 - cosine(self.reference_feature, feature)
            
            return max(0, similarity)  # 确保非负
        except Exception as e:
            print(f"匹配错误: {e}")
            return 0
    
    def run(self):
        """
        启动实时追踪（YOLOv8检测 + ResNet特征匹配）
        """
        # 配置RealSense相机
        pipeline = rs.pipeline()
        config = rs.config()
        
        print("正在启动相机...")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = pipeline.start(config)
        print("相机已启动！分辨率: 640x480@30fps")
        
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # 获取相机内参
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # 创建窗口
        window_name = 'Cat Tracking - YOLOv8 + ResNet'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print("\n开始实时追踪...")
        print("按 'q' 退出，按 's' 保存当前帧")
        
        frame_count = 0
        target_cat = None
        cats = []
        
        try:
            while True:
                # 获取帧
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                frame_count += 1
                
                # 每2帧检测一次（提高性能）
                if frame_count % 2 == 0:
                    print(f"处理第 {frame_count} 帧...", end='\r')
                    
                    # 使用YOLOv8检测猫
                    results = self.yolo_model(color_image, verbose=False)
                    
                    cats = []
                    if results[0].boxes is not None:
                        for detection in results[0].boxes:
                            # 获取类别ID和置信度
                            class_id = int(detection.cls)
                            confidence = float(detection.conf)
                            
                            # 类别15是cat
                            if class_id == 15 and confidence > 0.5:
                                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                                w = x2 - x1
                                h = y2 - y1
                                x = x1
                                y = y1
                                
                                # 确保边界在图像范围内
                                x = max(0, x)
                                y = max(0, y)
                                w = min(w, color_image.shape[1] - x)
                                h = min(h, color_image.shape[0] - y)
                                
                                if w > 30 and h > 30:  # 过滤太小的检测框
                                    cat_roi = color_image[y:y+h, x:x+w]
                                    
                                    # 使用ResNet特征比对
                                    match_score = self.match_cat(cat_roi)
                                    
                                    cats.append({
                                        'box': (x, y, w, h),
                                        'confidence': confidence,
                                        'match_score': match_score
                                    })
                    
                    # 找到最匹配的猫
                    target_cat = None
                    if cats:
                        # 排序显示所有猫的得分
                        sorted_cats = sorted(cats, key=lambda c: c['match_score'], reverse=True)
                        print(f"\n检测到 {len(cats)} 只猫 (YOLOv8):")
                        for i, cat in enumerate(sorted_cats[:3]):  # 显示前3名
                            print(f"  第{i+1}名: 相似度 {cat['match_score']:.4f}")
                        
                        target_cat = sorted_cats[0]
                        # ResNet相似度阈值0.75，差距要求0.05
                        if target_cat['match_score'] < 0.75:
                            print(f"  分数过低 ({target_cat['match_score']:.4f} < 0.75)")
                            target_cat = None
                        elif len(cats) > 1:
                            score_gap = target_cat['match_score'] - sorted_cats[1]['match_score']
                            if score_gap < 0.05:
                                print(f"  警告: 相似度差距太小 ({score_gap:.4f} < 0.05)，无法确定")
                                target_cat = None
                            else:
                                print(f"  ✓ 目标猫相似度 {target_cat['match_score']:.4f}，高于第二名 {score_gap:.4f}")
                
                # 绘制结果
                display_image = color_image.copy()
                
                if cats:
                    for cat in cats:
                        x, y, w, h = cat['box']
                        match_score = cat['match_score']
                        
                        if target_cat and cat == target_cat:
                            # 目标猫 - 绿色粗框
                            color = (0, 255, 0)
                            thickness = 3
                            
                            # 计算中心点的距离
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # 获取深度（米）
                            distance = depth_frame.get_distance(center_x, center_y)
                            
                            # 绘制框和标签
                            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, thickness)
                            
                            # 标注信息
                            label = f"TARGET Distance: {distance:.2f}m"
                            label2 = f"Similarity: {match_score:.3f}"
                            
                            # 背景框
                            cv2.rectangle(display_image, (x, y - 50), (x + 300, y), color, -1)
                            cv2.putText(display_image, label, (x + 5, y - 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            cv2.putText(display_image, label2, (x + 5, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            
                            # 在中心绘制十字准星
                            cv2.drawMarker(display_image, (center_x, center_y), 
                                         color, cv2.MARKER_CROSS, 20, 2)
                        else:
                            # 其他猫 - 灰色细框
                            color = (128, 128, 128)
                            thickness = 1
                            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, thickness)
                            cv2.putText(display_image, f"Other cat ({match_score:.3f})", 
                                      (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 显示提示信息
                cv2.putText(display_image, "Press 'q' to quit, 's' to save", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示相似度范围
                if cats:
                    max_score = max(cat['match_score'] for cat in cats)
                    cv2.putText(display_image, f"Max similarity: {max_score:.4f}", 
                              (10, display_image.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if target_cat is None and cats:
                    cv2.putText(display_image, "Target cat not found (low similarity)", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif target_cat is None:
                    cv2.putText(display_image, "No cat detected", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 显示图像
                cv2.imshow(window_name, display_image)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"tracked_cat_{frame_count}.jpg"
                    cv2.imwrite(filename, display_image)
                    print(f"已保存: {filename}")
        
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            print("追踪已停止")


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python3 track_specific_cat_yolov8.py <参考猫图片路径>")
        print("\n示例:")
        print("  python3 track_specific_cat_yolov8.py my_cat.jpg")
        print("\n说明:")
        print("  - 使用YOLOv8进行猫检测")
        print("  - 使用ResNet50进行深度特征提取")
        print("  - 余弦相似度进行精确匹配")
        sys.exit(1)
    
    reference_image_path = sys.argv[1]
    
    if not os.path.exists(reference_image_path):
        print(f"错误: 文件不存在 - {reference_image_path}")
        sys.exit(1)
    
    try:
        tracker = CatTracker(reference_image_path)
        tracker.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
