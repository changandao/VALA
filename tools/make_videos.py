import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def add_text_to_image(image_path, text, output_path, position='center_top'):
    """
    在图片上添加文字标注
    
    Args:
        image_path: 输入图片路径
        text: 要添加的文字
        output_path: 输出图片路径
        position: 文字位置，默认为中上位置
    """
    # 打开图片
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 尝试使用系统字体，如果没有则使用默认字体
    # 将字体在原有基础上放大 1.5 倍
    base_font_size = max(20, min(image.width, image.height) // 20)
    scaled_font_size = int(base_font_size * 1.5)
    font = None
    for font_name in ["times.ttf", "Times New Roman.ttf", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(font_name, scaled_font_size)
            break
        except:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 组装要显示的文字：添加 promt 和额外说明
    display_text = f"promt: {text}"

    # 获取文字尺寸
    if font:
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # 估算文字尺寸
        text_width = len(display_text) * 16
        text_height = 15
    
    # 计算文字位置（中上位置）
    x = (image.width - text_width) // 2
    y = image.height // 8  # 距离顶部1/8的位置
    
    # 添加文字背景（半透明黑色矩形）
    padding = max(10, (scaled_font_size if font else base_font_size) // 5)
    background_bbox = [
        x - padding, 
        y - padding, 
        x + text_width + padding, 
        y + text_height + padding
    ]
    
    # 创建一个带透明度的图层来绘制背景
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(background_bbox, fill=(0, 0, 0, 128))
    
    # 将overlay合并到原图
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    
    # 添加白色文字
    draw = ImageDraw.Draw(image)
    draw.text((x, y), display_text, fill='white', font=font)
    
    # 转换回RGB模式并保存
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image.save(output_path)

def process_folder(folder_path, text, start_num, end_num, output_folder):
    """
    处理一个文件夹中的图片，添加文字标注
    
    Args:
        folder_path: 输入文件夹路径
        text: 要添加的文字
        start_num: 开始的图片编号
        end_num: 结束的图片编号
        output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    processed_files = []
    
    for i in range(start_num, end_num + 1):
        # 构造文件名（假设格式为00001.jpg等）
        filename = f"{i:05d}.jpg"  # 5位数字，不足补0
        
        # 尝试不同的文件扩展名
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            input_path = os.path.join(folder_path, f"{i:05d}{ext}")
            if os.path.exists(input_path):
                output_path = os.path.join(output_folder, f"{i:05d}_annotated.jpg")
                
                try:
                    add_text_to_image(input_path, text, output_path)
                    processed_files.append((i, output_path))
                    print(f"处理完成: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"处理图片失败 {input_path}: {e}")
                break
        else:
            print(f"未找到图片: {i:05d}.* 在文件夹 {folder_path}")
    
    return processed_files

def create_video_from_images(image_list, output_video_path, fps=30):
    """
    从图片列表创建视频
    
    Args:
        image_list: 图片路径列表，按顺序排列
        output_video_path: 输出视频路径
        fps: 帧率
    """
    if not image_list:
        print("没有图片可以创建视频")
        return
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_list[0])
    if first_image is None:
        print(f"无法读取第一张图片: {image_list[0]}")
        return
    
    height, width, channels = first_image.shape
    
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 逐帧添加图片到视频
    for i, image_path in enumerate(image_list):
        image = cv2.imread(image_path)
        if image is not None:
            # 确保图片尺寸一致
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            video_writer.write(image)
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(image_list)} 帧")
        else:
            print(f"无法读取图片: {image_path}")
    
    # 释放VideoWriter对象
    video_writer.release()
    print(f"视频创建完成: {output_video_path}")

def copy_missing_images(backup_folder, missing_ranges, output_folder):
    """
    从备用文件夹复制缺失的图片
    
    Args:
        backup_folder: 备用图片文件夹路径
        missing_ranges: 缺失的编号范围列表
        output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    copied_files = []
    
    for start_num, end_num in missing_ranges:
        for i in range(start_num, end_num + 1):
            # 尝试不同的文件扩展名
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                backup_path = os.path.join(backup_folder, f"{i:05d}{ext}")
                if os.path.exists(backup_path):
                    output_path = os.path.join(output_folder, f"{i:05d}_backup.jpg")
                    
                    try:
                        # 直接复制图片（不添加文字标注）
                        image = Image.open(backup_path)
                        if image.mode == 'RGBA':
                            image = image.convert('RGB')
                        image.save(output_path)
                        copied_files.append((i, output_path))
                        print(f"复制备用图片: {backup_path} -> {output_path}")
                    except Exception as e:
                        print(f"复制图片失败 {backup_path}: {e}")
                    break
            else:
                print(f"未找到备用图片: {i:05d}.* 在文件夹 {backup_folder}")
    
    return copied_files

def main():
    # 定义文件夹路径和参数
    folders_config = [
        {
            'path': 'output/3dgs/waymo/1534950_1cams/supp_wopruning_wogate/train/renders_colormap_stair_test_3',
            'text': 'stair',
            'start': 7,
            'end': 39
        },
        {
            'path': 'output/3dgs/waymo/1534950_1cams/supp_wopruning_wogate/train/renders_colormap_car_test_3',
            'text': 'car',
            'start': 43,
            'end': 63
        },
        {
            'path': 'output/3dgs/waymo/1534950_1cams/supp_wopruning_wogate/train/renders_colormap_trash_bin_test_3',
            'text': 'trash_bin',
            'start': 70,
            'end': 113
        },
        {
            'path': 'output/3dgs/waymo/1534950_1cams/supp_wopruning_wogate/train/renders_colormap_tree_test_1',
            'text': 'tree',
            'start': 118,
            'end': 198
        }
    ]
    
    # 备用图片文件夹
    backup_folder = 'output/3dgs/waymo/1534950_1cams/train/ours_30000/renders'
    
    # 创建临时输出文件夹
    temp_output_folder = 'temp_annotated_images'
    os.makedirs(temp_output_folder, exist_ok=True)
    
    # 创建一个字典来存储所有图片（按编号索引）
    all_images_dict = {}
    
    # 处理每个文件夹
    for config in folders_config:
        print(f"\n处理文件夹: {config['path']}")
        print(f"添加文字: {config['text']}")
        print(f"图片范围: {config['start']:05d} - {config['end']:05d}")
        
        folder_output = os.path.join(temp_output_folder, config['text'])
        processed_files = process_folder(
            config['path'], 
            config['text'], 
            config['start'], 
            config['end'], 
            folder_output
        )
        
        # 将处理的文件添加到字典中
        for img_num, file_path in processed_files:
            all_images_dict[img_num] = file_path
    
    # 定义缺失的图片范围
    missing_ranges = [
        (0, 6),      # 00000-00006 (stair之前)
        (40, 42),    # 00040-00042 (stair和car之间)
        (64, 69),    # 00064-00069 (car和trash_bin之间)
        (114, 117)   # 00114-00117 (trash_bin和tree之间)
    ]
    
    # 复制缺失的图片
    print(f"\n从备用文件夹复制缺失的图片: {backup_folder}")
    backup_output = os.path.join(temp_output_folder, 'backup')
    backup_files = copy_missing_images(backup_folder, missing_ranges, backup_output)
    
    # 将备用图片添加到字典中
    for img_num, file_path in backup_files:
        all_images_dict[img_num] = file_path
    
    # 按编号顺序创建最终的图片列表
    all_processed_images = []
    for i in range(0, 199):  # 00000-00198
        if i in all_images_dict:
            all_processed_images.append(all_images_dict[i])
        else:
            print(f"警告: 缺失图片编号 {i:05d}")
    
    print(f"\n总计收集到 {len(all_processed_images)} 张图片")
    
    # 创建视频
    output_video = 'combined_annotation_video.mp4'
    print(f"\n开始创建视频，共 {len(all_processed_images)} 帧...")
    create_video_from_images(all_processed_images, output_video, fps=10)
    
    print(f"\n所有任务完成！")
    print(f"临时文件保存在: {temp_output_folder}")
    print(f"最终视频保存为: {output_video}")
    
    # 显示处理摘要
    print(f"\n处理摘要:")
    print(f"- 标注 'stair': 00007-00039 ({39-7+1} 张)")
    print(f"- 标注 'car': 00043-00063 ({63-43+1} 张)")
    print(f"- 标注 'trash_bin': 00070-00113 ({113-70+1} 张)")
    print(f"- 标注 'tree': 00118-00198 ({198-118+1} 张)")
    print(f"- 备用图片: 00000-00006, 00040-00042, 00064-00069, 00114-00117")
    print(f"- 总计: {len(all_processed_images)} 张图片")

if __name__ == "__main__":
    main()