"""
データセットを前処理する機能を提供します。
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import Brian2_Framework.Datasets as datasets

class Gabor_Filter:
    """
    ガボールフィルターを適用します。
    """
    def __init__(self, size:int, sigma:int, theta:int, lambd:int, gamma:int, psi:int):
        self.size = size
        self.sigma = sigma
        self.theta = theta
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        
    def _get_gabor_kernel(self):
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
        return cv2.getGaborKernel((self.size, self.size), self.sigma, self.theta, self.lambd, self.gamma, self.psi, ktype=cv2.CV_32F)
    
    def apply(self, image:np.ndarray):
        """
        ガボールフィルターを適用します。
        
        Args:
            image (np.ndarray): 画像
        Returns:
            np.ndarray: ガボールフィルターを適用した画像
        """
        kernel = self._get_gabor_kernel()
        return cv2.filter2D(image, cv2.CV_8UC3, kernel)
    
if __name__ == "__main__":
    img = datasets.get_mnist_image([8])[0][0]
    gabor = Gabor_Filter(size=11, sigma=2.201, theta=np.pi/2, lambd=5.6, gamma=1, psi=0)
    filtered_img = gabor.apply(img)
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(filtered_img)
    plt.figure()
    plt.imshow(gabor._get_gabor_kernel())
    plt.show()
