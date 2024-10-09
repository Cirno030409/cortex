import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

fig, ax = plt.subplots()

xlim = [0,100]
X, Y = [], []

def plot(frame):
    global X, Y
    
    Y.append(random.random())   # データを作成
    X.append(len(Y))
    
    if len(X) > 100:            # 描画範囲を更新
        xlim[0] += 1
        xlim[1] += 1
    
    ax.clear()                  # 前のグラフを削除
    line, = ax.plot(X, Y)       # 次のグラフを作成
    ax.set_title("sample animation (real time)")
    ax.set_ylim(-1, 2)
    ax.set_xlim(xlim[0], xlim[1])
    
    return [line]  # Artistオブジェクトのリストを返す

# 10msごとにplot関数を呼び出してアニメーションを作成
ani = animation.FuncAnimation(fig, plot, interval=10, blit=True)
plt.show(block=False)

while True:
    plt.pause(0.1)