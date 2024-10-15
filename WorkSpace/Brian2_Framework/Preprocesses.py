"""
データセットを前処理する機能を提供します。
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Gabor_Filter:
    """
    ガボールフィルターを適用します。
    """
        
    def _get_gabor_kernel(self, size, sigma, theta, lambd, gamma, psi):
        """
        ガボールカーネルを返します。
        
        Args:
            size (int): カーネルのサイズ
            sigma (int): ガウス分布の標準偏差
            theta (int): カーネルの角度
            lambd (int): 波長
            gamma (int): ガボールカーネルの形状(円形の度合い)
            psi (int): 位相シフト
        Returns:
            np.ndarray: ガボールカーネル
        """
        return cv2.getGaborKernel((size, size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    
    def apply2image(self, image:np.ndarray, size, sigma, theta, lambd, gamma, psi):
        """
        ガボールフィルターを適用します。
        
        Args:
            image (np.ndarray): 画像
            size (int): カーネルのサイズ
            sigma (int): ガウス分布の標準偏差
            theta (int): カーネルの角度
            lambd (int): 波長
            gamma (int): ガボールカーネルの形状(円形の度合い)
            psi (int): 位相シフト
        Returns:
            np.ndarray: ガボールフィルターを適用した画像
        """
        kernel = self._get_gabor_kernel(size, sigma, theta, lambd, gamma, psi)
        return cv2.filter2D(image, cv2.CV_8UC3, kernel)
    
if __name__ == "__main__":
    import Brian2_Framework.Datasets as datasets

    img = datasets.get_mnist_image([8])[0][0]
    gabor = Gabor_Filter(size=11, sigma=2.201, theta=np.pi/2, lambd=5.6, gamma=1, psi=0)
    filtered_img = gabor.apply(img, 11, 2.201, np.pi/2, 5.6, 1, 0)
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(filtered_img)
    plt.figure()
    plt.imshow(gabor)
    plt.show()
