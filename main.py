import cv2
import os
from core.transforms import PrivacyPreservingTransform


def main():
    input_path = "data/demo_input"
    output_path = "data/demo_output"
    os.makedirs(output_path, exist_ok=True)

    # 初始化脱敏器
    sanitizer = PrivacyPreservingTransform(epsilon=0.5)

    for img_name in os.listdir(input_path):
        img_raw = cv2.imread(os.path.join(input_path, img_name))
        if img_raw is None: continue

        # 处理结果（注意：Transform 内部会转为 Tensor，批量保存时我们转回 Numpy）
        processed_tensor = sanitizer(img_raw)
        processed_img = (processed_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')

        cv2.imwrite(os.path.join(output_path, img_name), processed_img)
        print(f"[*] 已完成脱敏: {img_name}")


if __name__ == "__main__":
    main()