import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os

class ScanNetColorMapper:
    """ScanNet数据集的颜色映射类 - 基于NYU40标签"""
    
    def __init__(self):
        # NYU40类别字典
        self.nyu40_dict = {
            0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
            6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
            11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
            16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
            21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
            26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
            31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
            36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
        }
        
        # 预测标签到ScanNet标签的映射
        self.target_mappings = {
            10: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 33],                                     # 10类
            15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],                  # 15类
            19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]   # 19类
        }
        
        # NYU40标签对应的标准颜色映射 (RGB值，范围0-255)
        self.scannet_colormap = {
            0: [0, 0, 0],         # unlabeled - 黑色
            1: [174, 199, 232],   # wall
            2: [152, 223, 138],   # floor
            3: [31, 119, 180],    # cabinet
            4: [255, 187, 120],   # bed
            5: [188, 189, 34],    # chair
            6: [140, 86, 75],     # sofa
            7: [255, 152, 150],   # table
            8: [214, 39, 40],     # door
            9: [197, 176, 213],   # window
            10: [148, 103, 189],  # bookshelf
            11: [196, 156, 148],  # picture
            12: [23, 190, 207],   # counter
            13: [178, 76, 76],    # blinds
            14: [247, 182, 210],  # desk
            15: [66, 188, 102],   # shelves
            16: [219, 219, 141],  # curtain
            17: [140, 57, 197],   # dresser
            18: [202, 185, 52],   # pillow
            19: [51, 176, 203],   # mirror
            20: [200, 54, 131],   # floormat
            21: [92, 193, 61],    # clothes
            22: [78, 71, 183],    # ceiling
            23: [172, 114, 82],   # books
            24: [255, 127, 14],   # refrigerator
            25: [91, 163, 138],   # television
            26: [153, 98, 156],   # paper
            27: [140, 153, 101],  # towel
            28: [158, 218, 229],  # showercurtain
            29: [100, 125, 154],  # box
            30: [178, 127, 135],  # whiteboard
            31: [120, 185, 128],  # person
            32: [146, 111, 194],  # nightstand
            33: [44, 160, 44],    # toilet
            34: [112, 128, 144],  # sink
            35: [96, 207, 209],   # lamp
            36: [227, 119, 194],  # bathtub
            37: [213, 92, 176],   # bag
            38: [94, 106, 211],   # otherstructure
            39: [82, 84, 163],    # otherfurniture
            40: [100, 85, 144],   # otherprop
        }
        
        # 默认颜色（用于未知标签）
        self.unknown_color = [128, 128, 128]  # 灰色表示未知类别
    
    def predict_id_to_scannet_id(self, predict_id: int, target_nums: int) -> int:
        """
        将预测的标签ID转换为ScanNet NYU40标签ID
        
        Args:
            predict_id: 预测的标签ID (从0开始)
            target_nums: 目标类别数量 (10, 15, 或 19)
            
        Returns:
            对应的ScanNet NYU40标签ID
        """
        if target_nums not in self.target_mappings:
            raise ValueError(f"不支持的target_nums: {target_nums}，支持的值: {list(self.target_mappings.keys())}")
        
        target_list = self.target_mappings[target_nums]
        
        if predict_id < 0 or predict_id >= len(target_list):
            print(f"警告: predict_id {predict_id} 超出范围 [0, {len(target_list)-1}]，返回unlabeled(0)")
            return 0  # 返回unlabeled
        
        return target_list[predict_id]
    
    def batch_predict_ids_to_scannet_ids(self, predict_ids: np.ndarray, target_nums: int) -> np.ndarray:
        """
        批量将预测标签ID转换为ScanNet NYU40标签ID
        
        Args:
            predict_ids: 预测标签ID数组
            target_nums: 目标类别数量 (10, 15, 或 19)
            
        Returns:
            ScanNet NYU40标签ID数组
        """
        scannet_ids = np.zeros_like(predict_ids, dtype=int)
        
        for i, pred_id in enumerate(predict_ids):
            scannet_ids[i] = self.predict_id_to_scannet_id(int(pred_id), target_nums)
        
        return scannet_ids

    def get_color(self, label_id: int) -> List[int]:
        """获取指定label_id对应的RGB颜色"""
        return self.scannet_colormap.get(label_id, self.unknown_color)
    
    def get_normalized_color(self, label_id: int) -> List[float]:
        """获取归一化的RGB颜色值（0-1范围）"""
        color = self.get_color(label_id)
        return [c / 255.0 for c in color]
    
    def get_class_name(self, label_id: int) -> str:
        """获取指定label_id对应的类别名称"""
        return self.nyu40_dict.get(label_id, "unknown")
    
    def labels_to_colors(self, labels: np.ndarray, normalized: bool = True) -> np.ndarray:
        """将标签数组转换为颜色数组"""
        colors = np.zeros((len(labels), 3))
        
        for i, label in enumerate(labels):
            if normalized:
                colors[i] = self.get_normalized_color(int(label))
            else:
                colors[i] = self.get_color(int(label))
        
        return colors
    
    def predict_labels_to_colors(self, predict_labels: np.ndarray, target_nums: int, 
                                normalized: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        将预测标签直接转换为颜色数组
        
        Args:
            predict_labels: 预测标签数组
            target_nums: 目标类别数量 (10, 15, 或 19)
            normalized: 是否返回归一化颜色
            
        Returns:
            (scannet_labels, colors): ScanNet标签数组和对应的颜色数组
        """
        # 先转换为ScanNet标签
        scannet_labels = self.batch_predict_ids_to_scannet_ids(predict_labels, target_nums)
        
        # 再转换为颜色
        colors = self.labels_to_colors(scannet_labels, normalized)
        
        return scannet_labels, colors
    
    # def predic_labels_to_normalized_colors(self, predict_labels: np.ndarray, target_nums: int) -> np.ndarray:
    #     """
    #     将预测标签直接转换为归一化颜色数组
    #     """
    #     scannet_labels = self.batch_predict_ids_to_scannet_ids(predict_labels, target_nums)
    #     colors = self.labels_to_colors(scannet_labels, normalized=True)
    #     return colors
    

def save_point_cloud_ply(mu, colors, output_path):
    """
    Save point cloud data as a PLY file.
    
    Args:
        mu (np.ndarray): Point cloud XYZ coordinates (N x 3)
        colors (np.ndarray): Point cloud RGB colors (N x 3)
        output_path (str): Full path to save the PLY file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Set points and colors
    pcd.points = o3d.utility.Vector3dVector(mu)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0,1]
    
    # Save point cloud
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")



class PointCloudProcessor:
    """点云处理类"""
    
    def __init__(self):
        self.color_mapper = ScanNetColorMapper()
    
    def load_pointcloud_from_array(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从numpy数组加载点云数据（ScanNet NYU40标签）
        
        Args:
            points: 形状为(N, 3)的点坐标数组
            labels: 形状为(N,)的ScanNet NYU40标签数组 (0-40)
            
        Returns:
            points, colors: 点坐标和对应的颜色
        """
        colors = self.color_mapper.labels_to_colors(labels, normalized=True)
        return points, colors
    
    def load_pointcloud_from_predictions(self, points: np.ndarray, predict_labels: np.ndarray, 
                                       target_nums: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从预测结果加载点云数据（需要转换为ScanNet标签）
        
        Args:
            points: 形状为(N, 3)的点坐标数组
            predict_labels: 形状为(N,)的预测标签数组 (从0开始)
            target_nums: 目标类别数量 (10, 15, 或 19)
            
        Returns:
            points, colors, scannet_labels: 点坐标、对应的颜色和转换后的ScanNet标签
        """
        scannet_labels, colors = self.color_mapper.predict_labels_to_colors(
            predict_labels, target_nums, normalized=True)
        return points, colors, scannet_labels
    
    def load_pointcloud_from_file(self, filepath: str, has_header: bool = False, 
                                is_prediction: bool = False, target_nums: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从文件加载点云数据
        
        Args:
            filepath: 文件路径
            has_header: 是否包含文件头
            is_prediction: 是否为预测结果（需要转换标签）
            target_nums: 如果is_prediction=True，需要指定目标类别数量
            
        Returns:
            points, colors: 点坐标和对应的颜色
        """
        # 读取数据
        skip_rows = 1 if has_header else 0
        data = np.loadtxt(filepath, skiprows=skip_rows)
        
        # 提取坐标和标签
        points = data[:, :3]  # x, y, z
        labels = data[:, 3].astype(int)  # label_id
        
        if is_prediction:
            if target_nums is None:
                raise ValueError("当is_prediction=True时，必须指定target_nums")
            # 转换预测标签为ScanNet标签
            scannet_labels, colors = self.color_mapper.predict_labels_to_colors(
                labels, target_nums, normalized=True)
            return points, colors
        else:
            # 直接使用ScanNet标签
            colors = self.color_mapper.labels_to_colors(labels, normalized=True)
            return points, colors
    
    def visualize_pointcloud(self, points: np.ndarray, colors: np.ndarray, 
                           window_name: str = "ScanNet Point Cloud"):
        """使用Open3D可视化点云"""
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd], window_name=window_name)
    
    def save_colored_pointcloud(self, points: np.ndarray, colors: np.ndarray, 
                              output_path: str):
        """保存带颜色的点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 保存为PLY格式
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"点云已保存到: {output_path}")
    
    def create_prediction_legend(self, target_nums: int, save_path: Optional[str] = None):
        """创建预测标签的颜色图例"""
        if target_nums not in self.color_mapper.target_mappings:
            raise ValueError(f"不支持的target_nums: {target_nums}")
        
        target_list = self.color_mapper.target_mappings[target_nums]
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(target_list) * 0.4)))
        
        # 绘制颜色条
        for i, scannet_id in enumerate(target_list):
            class_name = self.color_mapper.get_class_name(scannet_id)
            color = self.color_mapper.get_normalized_color(scannet_id)
            ax.barh(i, 1, color=color, height=0.8)
            ax.text(1.1, i, f"Pred ID {i} → ScanNet ID {scannet_id}: {class_name}", 
                   va='center', fontsize=9)
        
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.5, len(target_list) - 0.5)
        ax.set_xlabel('Color')
        ax.set_ylabel('Prediction Class ID')
        ax.set_title(f'Prediction to ScanNet Mapping ({target_nums} classes)')
        ax.set_yticks(range(len(target_list)))
        ax.set_yticklabels([str(i) for i in range(len(target_list))])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测标签颜色图例已保存到: {save_path}")
        
        plt.show()
    
    def create_color_legend(self, save_path: Optional[str] = None):
        """创建完整NYU40颜色图例"""
        fig, ax = plt.subplots(figsize=(12, 22))
        
        # 获取所有类别名称（按标签ID排序）
        sorted_labels = sorted(self.color_mapper.nyu40_dict.keys())
        
        # 绘制颜色条
        for i, label_id in enumerate(sorted_labels):
            class_name = self.color_mapper.nyu40_dict[label_id]
            color = self.color_mapper.get_normalized_color(label_id)
            ax.barh(i, 1, color=color, height=0.8)
            ax.text(1.1, i, f"{label_id}: {class_name}", va='center', fontsize=9)
        
        ax.set_xlim(0, 3.5)
        ax.set_ylim(-0.5, len(sorted_labels) - 0.5)
        ax.set_xlabel('Color')
        ax.set_ylabel('Class ID')
        ax.set_title('NYU40 ScanNet Color Map Legend')
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels([str(i) for i in sorted_labels])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"颜色图例已保存到: {save_path}")
        
        plt.show()
        
    def create_prediction_legend_horizontal(self, target_nums: int, save_path: Optional[str] = None):
        """创建预测标签的颜色图例（横向排列）"""
        if target_nums not in self.color_mapper.target_mappings:
            raise ValueError(f"不支持的target_nums: {target_nums}")
        
        target_list = self.color_mapper.target_mappings[target_nums]
        
        # 计算网格布局 - 每行最多显示8个色块
        cols_per_row = 8
        rows = (len(target_list) + cols_per_row - 1) // cols_per_row
        
        # 调整图像大小
        fig_width = min(16, len(target_list) * 1.8)
        fig_height = max(3, rows * 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 绘制颜色方块
        square_size = 0.8
        spacing = 1.0
        
        for i, scannet_id in enumerate(target_list):
            class_name = self.color_mapper.get_class_name(scannet_id)
            color = self.color_mapper.get_normalized_color(scannet_id)
            
            # 计算位置
            row = i // cols_per_row
            col = i % cols_per_row
            x = col * spacing
            y = (rows - 1 - row) * spacing  # 从上到下排列
            
            # 绘制颜色方块
            rect = plt.Rectangle((x, y), square_size, square_size, 
                               facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # 添加类别名称（在方块下方）
            ax.text(x + square_size/2, y - 0.15, class_name, 
                   ha='center', va='top', fontsize=9, rotation=0)
           
        
        # 设置坐标轴
        ax.set_xlim(-0.2, cols_per_row * spacing)
        ax.set_ylim(-0.5, rows * spacing)
        ax.set_aspect('equal')
        ax.axis('off')  # 隐藏坐标轴
        
        # 添加标题
        plt.suptitle(f'ScanNet Color Legend ({target_nums} classes)', 
                     fontsize=12, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测标签颜色图例已保存到: {save_path}")
        
    def create_color_legend(self, save_path: Optional[str] = None):
        """创建完整NYU40颜色图例"""
        fig, ax = plt.subplots(figsize=(12, 22))
        
        # 获取所有类别名称（按标签ID排序）
        sorted_labels = sorted(self.color_mapper.nyu40_dict.keys())
        
        # 绘制颜色条
        for i, label_id in enumerate(sorted_labels):
            class_name = self.color_mapper.nyu40_dict[label_id]
            color = self.color_mapper.get_normalized_color(label_id)
            ax.barh(i, 1, color=color, height=0.8)
            ax.text(1.1, i, f"{label_id}: {class_name}", va='center', fontsize=9)
        
        ax.set_xlim(0, 3.5)
        ax.set_ylim(-0.5, len(sorted_labels) - 0.5)
        ax.set_xlabel('Color')
        ax.set_ylabel('Class ID')
        ax.set_title('NYU40 ScanNet Color Map Legend')
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels([str(i) for i in sorted_labels])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"颜色图例已保存到: {save_path}")
        
        plt.show()
    
    def create_simple_color_legend(self, target_nums: int, save_path: Optional[str] = None):
        """创建简洁的颜色图例（完全按照用户图片样式）"""
        if target_nums not in self.color_mapper.target_mappings:
            raise ValueError(f"不支持的target_nums: {target_nums}")
        
        target_list = self.color_mapper.target_mappings[target_nums]
        target_list = [0, 1, 2, 3, 4, 7, 14, 16, 33, 12, 24, 34, 5, 11, 9, 8]
        
        # 每行显示的色块数量
        cols_per_row = len(target_list) if len(target_list) <= 10 else len(target_list)//2
        rows = (len(target_list) + cols_per_row - 1) // cols_per_row
        
        # 图像尺寸
        fig_width = cols_per_row * 1.5
        fig_height = rows * 1.2
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 绘制参数
        square_size = 0.8
        spacing_x = 2.1
        spacing_y = 0.5
        
        for i, scannet_id in enumerate(target_list):
            class_name = self.color_mapper.get_class_name(scannet_id)
            color = self.color_mapper.get_normalized_color(scannet_id)
            
            # 计算位置
            row = i // cols_per_row
            col = i % cols_per_row
            x = col * spacing_x
            if row==0:
                y=0.5
            else:   
                y=-0.05
            # y = (rows - 1 - row) * spacing_y
            
            # 绘制颜色方块
            rect = plt.Rectangle((x, y), 0.7, 0.5, 
                               facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            
            # 添加类别名称
            # ax.text(x + square_size + 0.1, y + square_size/2, class_name, 
            #        ha='left', va='center', fontsize=10, fontweight='normal')
            ax.text(x + 0.8, y+0.25, class_name, 
                   ha='left', va='center', fontsize=10, fontweight='normal')
        
        # 设置坐标轴
        max_x = cols_per_row * spacing_x
        max_y = rows * spacing_y
        ax.set_xlim(-0.1, max_x)
        ax.set_ylim(-0.2, max_y + 0.2)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"简洁颜色图例已保存到: {save_path}")
        
        plt.show()
        

def main():
    """主函数示例"""
    processor = PointCloudProcessor()
    
    # # 示例1: 使用ScanNet NYU40标签的数据
    # print("=== 示例1: ScanNet NYU40标签 ===")
    # n_points = 5000
    # points = np.random.randn(n_points, 3) * 2
    # scannet_labels = np.random.randint(0, 41, n_points)  # ScanNet标签 (0-40)
    
    # # 直接可视化ScanNet标签
    # points_colored, colors = processor.load_pointcloud_from_array(points, scannet_labels)
    # print("可视化ScanNet标签点云...")
    # processor.visualize_pointcloud(points_colored, colors, "ScanNet NYU40 Labels")
    
    # # 示例2: 使用预测标签的数据
    # print("\n=== 示例2: 预测标签转换 ===")
    # target_nums = 19  # 使用19类预测
    # predict_labels = np.random.randint(0, target_nums, n_points)  # 预测标签 (0-18)
    
    # # 转换预测标签为ScanNet标签并可视化
    # points_pred, colors_pred, converted_scannet_labels = processor.load_pointcloud_from_predictions(
    #     points, predict_labels, target_nums)
    # print(f"可视化{target_nums}类预测结果...")
    # processor.visualize_pointcloud(points_pred, colors_pred, f"Prediction {target_nums} Classes")
    
    # # 示例3: 单个标签转换测试
    # print("\n=== 示例3: 标签转换测试 ===")
    # color_mapper = processor.color_mapper
    
    # # 测试你提到的例子：预测ID 18 对应 ScanNet ID 36
    # test_predict_id = 18
    # test_target_nums = 19
    # converted_id = color_mapper.predict_id_to_scannet_id(test_predict_id, test_target_nums)
    # class_name = color_mapper.get_class_name(converted_id)
    # print(f"预测ID {test_predict_id} (19类) → ScanNet ID {converted_id} ({class_name})")
    
    # # 显示所有映射关系
    # for target_num in [10, 15, 19]:
    #     print(f"\n{target_num}类预测的映射关系:")
    #     target_list = color_mapper.target_mappings[target_num]
    #     for i, scannet_id in enumerate(target_list):
    #         class_name = color_mapper.get_class_name(scannet_id)
    #         print(f"  预测ID {i} → ScanNet ID {scannet_id} ({class_name})")
    
    # 创建预测标签的颜色图例
    # processor.create_prediction_legend(19, "prediction_19_classes_legend.png")
    processor.create_simple_color_legend(19, "prediction_19_classes_legend_simple.png")
    
    # 保存转换后的点云
    # processor.save_colored_pointcloud(points_pred, colors_pred, "prediction_colored_pointcloud.ply")
    
    # 示例4: 从文件加载预测结果
    print("\n=== 示例4: 文件加载示例 ===")
    # points, colors = processor.load_pointcloud_from_file(
    #     "your_prediction_file.txt", 
    #     is_prediction=True, 
    #     target_nums=19
    # )
    # processor.visualize_pointcloud(points, colors)

if __name__ == "__main__":
    main()

# 使用示例和测试
"""
=== 基本使用方法 ===

1. 直接使用ScanNet NYU40标签:
processor = PointCloudProcessor()
points, colors = processor.load_pointcloud_from_array(your_xyz, your_scannet_labels)
processor.visualize_pointcloud(points, colors)

2. 使用预测标签（需要转换）:
# 你的预测结果 predict_labels 范围是 0 到 target_nums-1
points, colors, scannet_labels = processor.load_pointcloud_from_predictions(
    your_xyz, your_predict_labels, target_nums=19)
processor.visualize_pointcloud(points, colors)

3. 单个标签转换:
color_mapper = ScanNetColorMapper()
scannet_id = color_mapper.predict_id_to_scannet_id(predict_id=18, target_nums=19)
# 结果: scannet_id = 36 (bathtub)

4. 批量标签转换:
scannet_labels = color_mapper.batch_predict_ids_to_scannet_ids(predict_array, target_nums=19)

5. 查看映射关系:
# 创建预测标签的颜色图例
processor.create_prediction_legend(target_nums=19)

=== 映射关系说明 ===
- 10类: predict_id 0-9  → scannet_id [1, 2, 4, 5, 6, 7, 8, 9, 10, 33]
- 15类: predict_id 0-14 → scannet_id [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]
- 19类: predict_id 0-18 → scannet_id [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]

=== 你的例子验证 ===
predict_id=18, target_nums=19 → scannet_id=36 (bathtub) ✓
"""


# 我现在有一个高斯点云，这个高斯点云中有xyz, 有label_id， 还有_features_dc这个attributes, 现在我希望新增一个函数能够把