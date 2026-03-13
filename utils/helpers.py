# utils/helpers.py
import matplotlib.pyplot as plt
import cv2
import os


def visualize_comparison(original_img_path, processed_img_path, save_path=None):
    """
    可视化对比工具：并排显示原始医学图像与隐私脱敏后的图像。
    """
    # 读取图像
    img1 = cv2.imread(original_img_path)
    img2 = cv2.imread(processed_img_path)

    if img1 is None or img2 is None:
        print("[-] 错误：无法读取图像文件，请检查路径。")
        return

    # OpenCV 默认读取为 BGR 格式，需要转为 RGB 以便 Matplotlib 正确显示色彩
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 绘制 1x2 的对比图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title('Sanitized Image (DP Noise Added)')
    axes[1].axis('off')

    plt.tight_layout()

    # 保存或展示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[*] 对比图已保存至: {save_path}")
    else:
        plt.show()