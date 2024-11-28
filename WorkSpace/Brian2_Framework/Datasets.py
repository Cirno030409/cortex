from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_mnist():
    """
    MNISTデータセットをダウンロードし、返します。

    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
        x_train, x_test: 画像データ (numpy配列)
        y_train, y_test: ラベルデータ (numpy配列)
    """
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    
    return (img_train, label_train), (img_test, label_test)

def get_mnist_sample(n_samples=1, dataset='train', labels:list=None):
    """
    MNISTデータセットからランダムにサンプルを取得します。

    Args:
        n_samples (int): 取得するサンプル数
        dataset (str): 'train'または'test'を指定
        labels (list, optional): 取得するラベルのリスト。Noneの場合は全ラベル(0-9)から取得

    Returns:
        tuple: (images, labels)
        images: 画像データ (numpy配列)
        labels: ラベルデータ (numpy配列)
    """
    (img_train, label_train), (img_test, label_test) = get_mnist()
    
    if dataset == 'train':
        img, label = img_train, label_train
    elif dataset == 'test':
        img, label = img_test, label_test
    else:
        raise ValueError("datasetは'train'または'test'を指定してください。")

    if labels is not None:
        # 指定されたラベルのインデックスのみを取得
        valid_indices = np.where(np.isin(label, labels))[0]
        img = img[valid_indices]
        label = label[valid_indices]
    
    try:
        indices = np.random.choice(len(img), n_samples, replace=False)
    except:
        raise ValueError("指定された枚数がMNISTデータセットの枚数を上回っている可能性があります。使用する枚数を再設定してください。")
    return img[indices], label[indices]

def get_mnist_sample_equality_labels(n_samples=1, dataset='train', labels:list=None):
    """
    各ラベルから均等枚数のランダム画像を取得します。
    
    Args:
        n_samples (int): 取得するサンプル数(指定ラベル数の倍数枚である必要がある)
        dataset (str): 'train'または'test'を指定
        labels (list, optional): 選択するラベルのリスト。Noneの場合は全ラベル(0-9)から選択

    Returns:
        tuple: (images, labels)
        images: 画像データ(numpy配列)
        labels: ラベルデータ(numpy配列)
    """
    if labels is None:
        labels = list(range(10))
    
    n_labels = len(labels)
    if n_samples % n_labels != 0:
        raise ValueError(f"n_samplesは{n_labels}の倍数である必要があります。")
    
    (img_train, label_train), (img_test, label_test) = get_mnist()
    if dataset == 'train':
        img, label_data = img_train, label_train
    elif dataset == 'test':
        img, label_data = img_test, label_test
    else:
        raise ValueError("datasetは'train'または'test'を指定してください。")
        
    imgs = []
    selected_labels = []
    n_img_per_label = n_samples // n_labels

    for label in labels:
        indices = np.where(label_data == label)[0]
        try:
            selected_indices = np.random.choice(indices, n_img_per_label, replace=False)
        except:
            raise ValueError("指定された枚数がMNISTデータセットの枚数を上回っている可能性があります。使用する枚数を再設定してください。")
        imgs.extend(img[selected_indices])
        selected_labels.extend([label] * n_img_per_label)
        
    # imgsとlabelsを対応を保ちながらシャッフルする
    combined = list(zip(imgs, selected_labels))
    np.random.shuffle(combined)
    imgs, selected_labels = zip(*combined)
        
    return np.array(imgs), np.array(selected_labels)

def get_mnist_image(labels, n_samples=1, down_sample=1, dataset='train'):
    """
    指定されたラベルからn_samples枚の画像をランダムに選択します。
    また、ダウンサンプリングを行うことも可能です。

    Args:
        labels (list): 選択するラベルのリスト
        n_samples (int, optional): 選択する画像の枚数
        down_sample (int, optional): ダウンサンプリングの倍率
        dataset (str, optional): 'train'または'test'を指定

    Raises:
        ValueError: datasetは'train'または'test'を指定してください。

    Returns:
        tuple: (images, labels)
        images: 画像データ(numpy配列)
        labels: ラベルデータ(numpy配列)
    """
    (img_train, label_train), (img_test, label_test) = get_mnist()
    if dataset == 'train':
        img, label_data = img_train, label_train
    elif dataset == 'test':
        img, label_data = img_test, label_test
    else:
        raise ValueError("datasetは'train'または'test'を指定してください。")
    
    selected_images = []
    selected_labels = []
    
    for label in labels:
        indices = np.where(label_data == label)[0]
        
        # 各ラベルに対してn_samples枚の画像をランダムに選択
        try:
            label_indices = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)
        except:
            raise ValueError("指定された枚数がMNISTデータセットの枚数を上回っている可能性があります。使用する枚数を再設定してください。")
        selected_images.extend(img[label_indices])
        selected_labels.extend(label_data[label_indices])
    
    # 選択された画像とラベルをシャッフル
    combined = list(zip(selected_images, selected_labels))
    np.random.shuffle(combined)
    selected_images, selected_labels = zip(*combined)
    
    selected_images = np.array(selected_images)
    selected_labels = np.array(selected_labels)

    # ダウンサンプリングの実装
    if down_sample > 1:
        new_size = selected_images.shape[1] // down_sample
        downsampled_images = np.zeros((len(selected_images), new_size, new_size))
        for idx, img in enumerate(selected_images):
            for i in range(0, img.shape[0], down_sample):
                for j in range(0, img.shape[1], down_sample):
                    downsampled_images[idx, i//down_sample, j//down_sample] = np.max(img[i:i+down_sample, j:j+down_sample])
        return downsampled_images, selected_labels
    return selected_images, selected_labels

def show_image(images:list, labels:list=None, save_path:str=None) -> None:
    """
    提供されたMnist画像を表示します。

    Args:
        images (list): Mnist画像
        labels (list): Mnistラベル
        save_path (str): 保存するパス
    """
    plt.figure(figsize=(15, 9))
    n_images = len(images)
    n_cols = 10
    n_rows = (n_images + n_cols - 1) // n_cols
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        if labels is not None:
            plt.title(labels[i])
        plt.axis('off')
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    
def save_images(images:list, save_path:str):
    """
    提供された画像をすべて個別の画像として保存します。

    Args:
        images (list): 画像リスト
    """
    for i in tqdm(range(len(images))):
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.savefig(f"{save_path}/image_{i}.png")
        plt.close()
    
def divide_image_into_chunks(image, chunk_size):
    """
    一枚の画像データを指定されたサイズのチャンクに分割して、チャンク画像リストを返します。チャンクサイズは画像の約数である必要があります。

    Args:
        images (numpy.ndarray): 画像データ
        chunk_size (int): チャンクのサイズ
    """
    # images[28, 28]
    n_chunks = image.shape[0] // chunk_size
    chunks = []
    for i in range(n_chunks):
        for j in range(n_chunks):
            chunk = image[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size]
            chunks.append(chunk)
    return np.array(chunks)

if __name__ == "__main__":
    images, labels = get_mnist_sample_equality_labels(n_samples=30, dataset='train', labels=[0, 1, 2])
    print(labels)
    
    
    
    
    
    
