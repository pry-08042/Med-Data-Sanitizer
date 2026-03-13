import cv2
import numpy as np
import torch


class PrivacyPreservingTransform:
    """
    符合 PyTorch 规范的医学图像隐私保护转换类。
    集成视觉脱敏与差分隐私噪声注入。
    """

    def __init__(self, epsilon=0.5, target_size=(256, 256), apply_dp=True):
        self.epsilon = epsilon
        self.target_size = target_size
        self.apply_dp = apply_dp

    def _crop_artifacts(self, image):
        """自动化裁剪：消除医学图像中的黑边或标尺标记"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            image = image[y:y + h, x:x + w]

        # 统一缩放到模型输入尺寸
        return cv2.resize(image, self.target_size)

    def _add_dp_noise(self, image):
        """差分隐私：向像素注入拉普拉斯噪声"""
        if not self.apply_dp:
            return image

        scale = 1.0 / self.epsilon
        noise = np.random.laplace(0, scale, image.shape)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def __call__(self, image):
        """
        支持 PyTorch DataLoader 调用。
        输入可以是 Numpy 数组，输出为脱敏后的 Tensor。
        """
        # 1. 视觉特征脱敏
        image = self._crop_artifacts(image)

        # 2. 隐私噪声注入
        image = self._add_dp_noise(image)

        # 3. 转换为 Tensor 并归一化 (兼容 PyTorch 训练)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image_tensor