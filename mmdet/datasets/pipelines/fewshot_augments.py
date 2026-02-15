# mmdet/datasets/pipelines/fewshot_augments.py

# mmdet/datasets/pipelines/fewshot_augments.py

import numpy as np
import cv2
from ..builder import PIPELINES
import mmcv


@PIPELINES.register_module()
class fs_MultiScaleResize(object):
    """多尺度随机缩放增强，用于处理不同大小的目标
    
    Args:
        scale_range (tuple): 缩放范围，如(0.5, 1.5)表示缩放到原图的50%-150%
        base_size (tuple): 基础图像大小，默认(800, 800)
        keep_ratio (bool): 是否保持长宽比
    """
    
    def __init__(self, 
                 scale_range=(0.5, 1.5),
                 base_size=(800, 800),
                 keep_ratio=True):
        self.scale_range = scale_range
        self.base_size = base_size
        self.keep_ratio = keep_ratio
        
    def __call__(self, results):
        """随机选择一个缩放比例并调整图像大小"""
        # 随机选择缩放比例
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        # 计算目标尺寸
        target_size = (int(self.base_size[0] * scale_factor), 
                      int(self.base_size[1] * scale_factor))
        
        img = results['img']
        h, w = img.shape[:2]
        
        if self.keep_ratio:
            # 保持长宽比的缩放
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = target_size
            scale = np.array([new_w / w, new_h / h])
        
        # 缩放图像
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        results['img'] = resized_img
        results['img_shape'] = resized_img.shape
        
        # 缩放边界框 - 确保边界框随图像正确缩放
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            bboxes = results['gt_bboxes'].copy()  # 复制以避免修改原始数据
            if self.keep_ratio:
                # 等比例缩放：x1, y1, x2, y2都乘以相同的scale
                bboxes = bboxes * scale
            else:
                # 非等比例缩放：x坐标乘以x_scale，y坐标乘以y_scale
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale[0]
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale[1]
            results['gt_bboxes'] = bboxes
            
        # 更新scale_factor
        if self.keep_ratio:
            results['scale_factor'] = np.array([scale, scale, scale, scale], dtype=np.float32)
        else:
            results['scale_factor'] = np.array([scale[0], scale[1], scale[0], scale[1]], dtype=np.float32)
            
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale_range={self.scale_range}, '
        repr_str += f'base_size={self.base_size}, '
        repr_str += f'keep_ratio={self.keep_ratio})'
        return repr_str


@PIPELINES.register_module()
class fs_RandomRotate(object):
    """随机旋转增强，适用于遥感图像
    
    Args:
        angles (list): 可选的旋转角度列表，如[0, 90, 180, 270]
        prob (float): 应用旋转的概率
        border_value (int): 边界填充值
        center (tuple): 旋转中心，None表示图像中心
    """
    
    def __init__(self, 
                 angles=[0, 90, 180, 270],
                 prob=0.5,
                 border_value=0,
                 center=None):
        self.angles = angles
        self.prob = prob
        self.border_value = border_value
        self.center = center
        
    def _rotate_bbox(self, bboxes, rotate_matrix, img_shape):
        """旋转边界框
        
        Args:
            bboxes: 边界框坐标 (n, 4) [x1, y1, x2, y2]
            rotate_matrix: 旋转矩阵
            img_shape: 图像形状
            
        Returns:
            rotated_bboxes: 旋转后的边界框
        """
        h, w = img_shape[:2]
        
        # 获取每个bbox的四个角点
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        
        corners = np.array([
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x2, y2],  # 右下
            [x1, y2]   # 左下
        ])  # shape: (4, n, 2)
        
        # 转换为齐次坐标
        num_bboxes = bboxes.shape[0]
        corners_homo = np.ones((4 * num_bboxes, 3))
        for i in range(4):
            corners_homo[i*num_bboxes:(i+1)*num_bboxes, :2] = corners[i].T
        
        # 应用旋转
        rotated_corners = rotate_matrix @ corners_homo.T  # (2, 4*n)
        rotated_corners = rotated_corners.T.reshape(4, num_bboxes, 2)
        
        # 获取新的边界框（包围旋转后的四个角点）
        new_x1 = np.min(rotated_corners[:, :, 0], axis=0)
        new_y1 = np.min(rotated_corners[:, :, 1], axis=0)
        new_x2 = np.max(rotated_corners[:, :, 0], axis=0)
        new_y2 = np.max(rotated_corners[:, :, 1], axis=0)
        
        # 裁剪到图像边界内
        new_x1 = np.clip(new_x1, 0, w)
        new_y1 = np.clip(new_y1, 0, h)
        new_x2 = np.clip(new_x2, 0, w)
        new_y2 = np.clip(new_y2, 0, h)
        
        rotated_bboxes = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
        
        # 过滤掉无效的bbox（面积太小的）
        areas = (new_x2 - new_x1) * (new_y2 - new_y1)
        valid_inds = areas > 1  # 面积大于1的保留
        
        return rotated_bboxes, valid_inds
        
    def __call__(self, results):
        """应用随机旋转"""
        if np.random.rand() > self.prob:
            return results
            
        # 随机选择旋转角度
        angle = np.random.choice(self.angles)
        if angle == 0:
            return results
            
        img = results['img']
        h, w = img.shape[:2]
        
        # 确定旋转中心
        if self.center is None:
            center = (w // 2, h // 2)
        else:
            center = self.center
            
        # 获取旋转矩阵
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 对于90度的倍数，需要调整输出图像大小
        if angle % 90 == 0:
            if angle % 180 == 90:
                # 90度或270度旋转，交换宽高
                new_w, new_h = h, w
                # 调整旋转矩阵的平移部分
                if angle == 90:
                    rotate_matrix[0, 2] = new_w // 2 - center[1]
                    rotate_matrix[1, 2] = new_h // 2 + center[0] - w
                else:  # 270度
                    rotate_matrix[0, 2] = new_w // 2 + center[1] - h
                    rotate_matrix[1, 2] = new_h // 2 - center[0]
            else:
                # 180度旋转，保持原大小
                new_w, new_h = w, h
        else:
            # 非90度倍数的旋转，计算包围框
            cos = np.abs(rotate_matrix[0, 0])
            sin = np.abs(rotate_matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            # 调整旋转矩阵
            rotate_matrix[0, 2] += (new_w - w) / 2
            rotate_matrix[1, 2] += (new_h - h) / 2
            
        # 旋转图像
        rotated_img = cv2.warpAffine(
            img, rotate_matrix, (new_w, new_h), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value
        )
        
        results['img'] = rotated_img
        results['img_shape'] = rotated_img.shape
        
        # 旋转边界框
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            rotated_bboxes, valid_inds = self._rotate_bbox(
                results['gt_bboxes'], rotate_matrix, (new_h, new_w)
            )
            results['gt_bboxes'] = rotated_bboxes[valid_inds]
            
            # 同步更新标签
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]
                
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angles={self.angles}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'border_value={self.border_value})'
        return repr_str


@PIPELINES.register_module()
class fs_RandomFlipRotate(object):
    """组合的翻转和旋转增强，专门用于遥感图像
    
    这个类结合了水平翻转、垂直翻转和旋转，提供更丰富的增强效果
    
    Args:
        flip_ratio (float): 翻转概率
        rotate_ratio (float): 旋转概率
        angles (list): 可选的旋转角度
    """
    
    def __init__(self,
                 flip_ratio=0.5,
                 rotate_ratio=0.5,
                 angles=[90, 180, 270]):
        self.flip_ratio = flip_ratio
        self.rotate_ratio = rotate_ratio
        self.angles = angles
        
    def __call__(self, results):
        # 随机水平翻转
        if np.random.rand() < self.flip_ratio:
            results = self._flip(results, direction='horizontal')
            
        # 随机垂直翻转
        if np.random.rand() < self.flip_ratio:
            results = self._flip(results, direction='vertical')
            
        # 随机旋转
        if np.random.rand() < self.rotate_ratio:
            angle = np.random.choice(self.angles)
            if angle != 0:
                results = self._rotate(results, angle)
                
        return results
    
    def _flip(self, results, direction='horizontal'):
        """翻转图像和边界框"""
        img = results['img']
        h, w = img.shape[:2]
        
        if direction == 'horizontal':
            img = np.flip(img, axis=1)
            if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
                bboxes = results['gt_bboxes'].copy()
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
                results['gt_bboxes'] = bboxes
        else:  # vertical
            img = np.flip(img, axis=0)
            if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
                bboxes = results['gt_bboxes'].copy()
                bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
                results['gt_bboxes'] = bboxes
                
        results['img'] = np.ascontiguousarray(img)
        return results
    
    def _rotate(self, results, angle):
        """简化的90度倍数旋转"""
        img = results['img']
        h, w = img.shape[:2]
        
        # 旋转图像
        if angle == 90:
            img = np.rot90(img, k=1, axes=(0, 1))
        elif angle == 180:
            img = np.rot90(img, k=2, axes=(0, 1))
        elif angle == 270:
            img = np.rot90(img, k=3, axes=(0, 1))
            
        results['img'] = np.ascontiguousarray(img)
        results['img_shape'] = img.shape
        
        # 旋转边界框
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            bboxes = results['gt_bboxes'].copy()
            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            
            if angle == 90:
                new_x1 = y1
                new_y1 = w - x2
                new_x2 = y2
                new_y2 = w - x1
                new_w, new_h = h, w
            elif angle == 180:
                new_x1 = w - x2
                new_y1 = h - y2
                new_x2 = w - x1
                new_y2 = h - y1
                new_w, new_h = w, h
            elif angle == 270:
                new_x1 = h - y2
                new_y1 = x1
                new_x2 = h - y1
                new_y2 = x2
                new_w, new_h = h, w
                
            results['gt_bboxes'] = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
            
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(flip_ratio={self.flip_ratio}, '
        repr_str += f'rotate_ratio={self.rotate_ratio}, '
        repr_str += f'angles={self.angles})'
        return repr_str