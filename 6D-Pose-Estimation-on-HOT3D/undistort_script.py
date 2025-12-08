import os
import json
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict

# ==================================================================================
#                                 I. 核心配置区域
# ==================================================================================

# --- 1. 数据集根目录 ---
# 请将此路径设置为您截图所示的那个文件夹的路径
# 即 'gray1', 'gray2', 'mask_gray1' 等文件夹所在的父目录。
# 示例: r"C:\Users\YourName\Desktop\fisheye_data"
BASE_DIR = Path(r"")  # <--- 请修改为您的实际路径

# --- 2. 待处理的数据集标识 ---
# 脚本将查找与这些标识相关的文件，例如 'gray1', 'mask_gray1', 'scene_camera_gray1.json'
SETS_TO_PROCESS = ["gray1", "gray2"]

# --- 3. 叠加验证图的颜色和透明度设置 (BGR格式) ---
COLOR_FULL_MASK = (0, 0, 255)  # 红色: 完整物体的 Mask 和 BBox
COLOR_VISIB_MASK = (0, 255, 0)  # 绿色: 可见部分的 Mask 和 BBox
ALPHA_FULL = 0.3  # 完整 Mask 的透明度
ALPHA_VISIB = 0.5  # 可见 Mask 的透明度


# ==================================================================================
#                                 II. 辅助函数
# ==================================================================================

def calculate_mask_info(mask_image: np.ndarray) -> Tuple[List[int], int]:
    """
    计算给定二值 Mask 的边界框 [x, y, width, height] 和非零像素数量。
    """
    coords = np.argwhere(mask_image > 0)
    if coords.size == 0:
        return [0, 0, 0, 0], 0  # Mask 为空

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    px_count = int(np.count_nonzero(mask_image))
    return bbox, px_count


# ==================================================================================
#                               III. 核心处理函数
# ==================================================================================

def process_fisheye_set(base_dir: Path, set_name: str):
    """
    对指定的数据集集合（如 "gray1" 或 "gray2"）进行完整的去畸变处理。
    """
    print(f"\n{'=' * 30}\n--- 开始处理数据集: {set_name} ---\n{'=' * 30}")
    start_time = time.time()

    # --- 1. 构建输入和输出路径 ---
    # 输入路径
    input_rgb_dir = base_dir / set_name
    input_mask_dir = base_dir / f"mask_{set_name}"
    input_mask_visib_dir = base_dir / f"mask_visib_{set_name}"
    input_cam_json = base_dir / f"scene_camera_{set_name}.json"
    input_gt_json = base_dir / f"scene_gt_{set_name}.json"

    # 输出路径 (在原名后加 _undistorted)
    output_rgb_dir = base_dir / f"{set_name}_undistorted"
    output_mask_dir = base_dir / f"mask_{set_name}_undistorted"
    output_mask_visib_dir = base_dir / f"mask_visib_{set_name}_undistorted"
    output_gt_info_json = base_dir / f"scene_gt_info_{set_name}_undistorted.json"
    output_overlay_dir = base_dir / "undistorted_overlay"  # 验证图统一存放

    # --- 2. 创建输出目录 ---
    print("--- 步骤 1/4: 创建输出目录 ---")
    output_rgb_dir.mkdir(exist_ok=True)
    output_mask_dir.mkdir(exist_ok=True)
    output_mask_visib_dir.mkdir(exist_ok=True)
    output_overlay_dir.mkdir(exist_ok=True)
    print(f"  - RGB输出到: {output_rgb_dir.name}")
    print(f"  - Mask输出到: {output_mask_dir.name}")
    print(f"  - 可见Mask输出到: {output_mask_visib_dir.name}")
    print(f"  - 验证图输出到: {output_overlay_dir.name}")

    # --- 3. 加载场景级 JSON 文件 ---
    print("\n--- 步骤 2/4: 加载JSON文件 ---")
    try:
        with open(input_cam_json, 'r', encoding='utf-8') as f:
            cam_data = json.load(f)
        with open(input_gt_json, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        print(f"  - 成功加载相机文件: {input_cam_json.name}")
        print(f"  - 成功加载GT文件: {input_gt_json.name}")
    except FileNotFoundError as e:
        print(f"  ❌ 错误: 缺少必要的JSON文件: {e.filename}。跳过处理 '{set_name}'。")
        return
    except json.JSONDecodeError as e:
        print(f"  ❌ 错误: JSON文件格式错误: {e}。跳过处理 '{set_name}'。")
        return

    # --- 4. 遍历所有图像进行去畸变处理 ---
    print("\n--- 步骤 3/4: 开始批量去畸变图像和Mask ---")

    all_new_gt_info: Dict[str, List] = {}  # 存储新生成的scene_gt_info
    image_ids_to_process = sorted(gt_data.keys(), key=int)
    total_images = len(image_ids_to_process)

    for i, im_id_str in enumerate(image_ids_to_process):
        im_id = int(im_id_str)
        print(f"\r  处理图像: {im_id:06d} ({i + 1}/{total_images})...", end='', flush=True)

        # 获取相机参数
        cam_info = cam_data.get(im_id_str)
        if not cam_info:
            print(f"\n  ⚠️ 警告: 图像 {im_id:06d} 在 {input_cam_json.name} 中缺少相机信息，跳过。")
            continue

        proj_params = cam_info["cam_model"]["projection_params"]

        # 原始相机内参 K_orig 和畸变系数 D_orig (HOT3D FISHEYE624 格式)
        fx_orig, cx_orig, cy_orig = proj_params[0], proj_params[1], proj_params[2]
        K_orig = np.array([[fx_orig, 0, cx_orig], [0, fx_orig, cy_orig], [0, 0, 1]], dtype=np.float32)
        D_orig = np.array(proj_params[4:8], dtype=np.float32)  # [k1, k2, k3, k4]

        # 去畸变后的目标相机内参 K_new (创建一个标准的针孔相机矩阵)
        K_new = K_orig.copy()

        # 读取原始RGB图像
        img_orig_path = input_rgb_dir / f"{im_id:06d}.jpg"
        img_orig = cv2.imread(str(img_orig_path))
        if img_orig is None:
            print(f"\n  ⚠️ 警告: 无法加载图像 {img_orig_path}，跳过。")
            continue

        h, w = img_orig.shape[:2]

        # 计算去畸变映射
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_orig, D_orig, np.eye(3), K_new, (w, h), cv2.CV_32FC1)

        # 去畸变并保存RGB图像
        undistorted_rgb = cv2.remap(img_orig, map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(str(output_rgb_dir / f"{im_id:06d}.jpg"), undistorted_rgb)

        # 准备处理当前图像的所有物体实例
        current_image_gt_info = []
        # 用于叠加验证的合并Mask
        combined_mask_full = np.zeros((h, w), dtype=np.uint8)
        combined_mask_visib = np.zeros((h, w), dtype=np.uint8)

        gt_instances = gt_data.get(im_id_str, [])
        for inst_idx, gt_instance in enumerate(gt_instances):
            inst_id_padded = f"{inst_idx:06d}"

            # 去畸变并保存完整Mask
            mask_orig_path = input_mask_dir / f"{im_id:06d}_{inst_id_padded}.png"
            mask_orig = cv2.imread(str(mask_orig_path), cv2.IMREAD_GRAYSCALE)
            if mask_orig is not None:
                undistorted_mask = cv2.remap(mask_orig, map1, map2, cv2.INTER_NEAREST)
                undistorted_mask = (undistorted_mask > 0).astype(np.uint8) * 255
                cv2.imwrite(str(output_mask_dir / f"{im_id:06d}_{inst_id_padded}.png"), undistorted_mask)
            else:
                undistorted_mask = np.zeros((h, w), dtype=np.uint8)  # 如果mask不存在则创建空mask

            # 去畸变并保存可见Mask
            mask_visib_orig_path = input_mask_visib_dir / f"{im_id:06d}_{inst_id_padded}.png"
            mask_visib_orig = cv2.imread(str(mask_visib_orig_path), cv2.IMREAD_GRAYSCALE)
            if mask_visib_orig is not None:
                undistorted_mask_visib = cv2.remap(mask_visib_orig, map1, map2, cv2.INTER_NEAREST)
                undistorted_mask_visib = (undistorted_mask_visib > 0).astype(np.uint8) * 255
                cv2.imwrite(str(output_mask_visib_dir / f"{im_id:06d}_{inst_id_padded}.png"), undistorted_mask_visib)
            else:
                undistorted_mask_visib = np.zeros((h, w), dtype=np.uint8)

            # 计算新GT信息
            bbox_obj, px_count_all = calculate_mask_info(undistorted_mask)
            bbox_visib, px_count_visib = calculate_mask_info(undistorted_mask_visib)
            visib_fract = px_count_visib / px_count_all if px_count_all > 0 else 0.0

            current_image_gt_info.append({
                "obj_id": gt_instance["obj_id"],
                "bbox_obj": bbox_obj,
                "bbox_visib": bbox_visib,
                "px_count_all": px_count_all,
                "px_count_visib": px_count_visib,
                "visib_fract": float(visib_fract)
            })

            # 合并mask用于验证图
            combined_mask_full = cv2.bitwise_or(combined_mask_full, undistorted_mask)
            combined_mask_visib = cv2.bitwise_or(combined_mask_visib, undistorted_mask_visib)

        all_new_gt_info[im_id_str] = current_image_gt_info

        # 生成并保存验证图
        overlay_img = undistorted_rgb.copy()
        # 叠加红色完整Mask
        overlay_red = np.zeros_like(overlay_img)
        overlay_red[combined_mask_full > 0] = COLOR_FULL_MASK
        overlay_img = cv2.addWeighted(overlay_img, 1.0, overlay_red, ALPHA_FULL, 0)
        # 叠加绿色可见Mask
        overlay_green = np.zeros_like(overlay_img)
        overlay_green[combined_mask_visib > 0] = COLOR_VISIB_MASK
        overlay_img = cv2.addWeighted(overlay_img, 1.0, overlay_green, ALPHA_VISIB, 0)

        # 绘制BBox
        for info in current_image_gt_info:
            if info["bbox_obj"][2] > 0:
                x, y, w, h = info["bbox_obj"]
                cv2.rectangle(overlay_img, (x, y), (x + w, y + h), COLOR_FULL_MASK, 1)
            if info["bbox_visib"][2] > 0:
                x, y, w, h = info["bbox_visib"]
                cv2.rectangle(overlay_img, (x, y), (x + w, y + h), COLOR_VISIB_MASK, 1)

        # 保存验证图时，在文件名中加入set_name以区分
        cv2.imwrite(str(output_overlay_dir / f"{set_name}_{im_id:06d}.jpg"), overlay_img)

    print(f"\n  去畸变处理完成。")

    # --- 5. 保存新生成的 scene_gt_info 文件 ---
    print("\n--- 步骤 4/4: 保存新生成的 GT Info 文件 ---")
    with open(output_gt_info_json, 'w', encoding='utf-8') as f:
        json.dump(all_new_gt_info, f, indent=4)
    print(f"  - 成功生成并保存: {output_gt_info_json.name}")

    end_time = time.time()
    print(f"\n--- 数据集 '{set_name}' 处理完成！总耗时: {end_time - start_time:.2f} 秒 ---")


# ==================================================================================
#                                 IV. 主程序入口
# ==================================================================================

if __name__ == "__main__":
    print("==========================================================================")
    print("===      鱼眼数据集原地去畸变处理脚本 (Undistortion Script)      ===")
    print("==========================================================================")
    print(f"  将处理以下数据集标识: {', '.join(SETS_TO_PROCESS)}")
    print(f"  数据根目录: {BASE_DIR}")
    print("  所有生成的文件/文件夹将带有 '_undistorted' 后缀并存放在同一目录。")
    print("--------------------------------------------------------------------------")
    input("  按 Enter 键开始处理，或按 Ctrl+C 取消...")

    for set_id in SETS_TO_PROCESS:
        process_fisheye_set(BASE_DIR, set_id)

    print("\n==========================================================================")
    print("===                      所有处理任务已完成！                      ===")
    print("==========================================================================")

