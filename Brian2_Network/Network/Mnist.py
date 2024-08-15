from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def load_mnist():
    """
    MNISTデータセットをダウンロードし、提供します。

    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
        x_train, x_test: 画像データ (numpy配列)
        y_train, y_test: ラベルデータ (numpy配列)
    """
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    
    
    print("MNISTデータセットの読み込みが完了しました。")
    print(f"訓練データ: {img_train.shape[0]}枚")
    print(f"テストデータ: {img_test.shape[0]}枚")
    
    return (img_train, label_train), (img_test, label_test)

def get_mnist_sample(n_samples=1, dataset='train'):
    """
    MNISTデータセットからランダムにサンプルを取得します。

    Args:
        n_samples (int): 取得するサンプル数
        dataset (str): 'train'または'test'を指定

    Returns:
        tuple: (images, labels)
        images: 画像データ (numpy配列)
        labels: ラベルデータ (numpy配列)
    """
    (img_train, label_train), (img_test, label_test) = load_mnist()
    
    if dataset == 'train':
        img, label = img_train, label_train
    elif dataset == 'test':
        img, label = img_test, label_test
    else:
        raise ValueError("datasetは'train'または'test'を指定してください。")
    
    indices = np.random.choice(len(img), n_samples, replace=False)
    return img[indices], label[indices]

def get_mnist_image(label, n_samples=1, down_sample=1, dataset='train'):
    (img_train, label_train), (img_test, label_test) = load_mnist()
    if dataset == 'train':
        img, label = img_train, label_train
    elif dataset == 'test':
        img, label = img_test, label_test
    else:
        raise ValueError("datasetは'train'または'test'を指定してください。")
    indices = np.where(label == label)[0]
    indices = np.random.choice(indices, n_samples, replace=False)
    
    # ダウンサンプリングの実装
    if down_sample > 1:
        new_size = img.shape[1] // down_sample
        downsampled_images = np.zeros((len(indices), new_size, new_size))
        for idx, img in enumerate(img[indices]):
            for i in range(0, img.shape[1], down_sample):
                for j in range(0, img.shape[2], down_sample):
                    downsampled_images[idx, i//down_sample, j//down_sample] = np.max(img[i:i+down_sample, j:j+down_sample])
        return downsampled_images, label[indices]
    return img[indices], label[indices]

if __name__ == "__main__":
    images, labels = get_mnist_image(label=3, n_samples=20, down_sample=2, dataset='train')
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.show()
