import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
import os
import glob
from matplotlib.animation import FuncAnimation
import Brian2_Framework.Tools as tools
import seaborn as sns
import matplotlib.animation as animation
import random
import mplcursors
from collections import deque
import time
import mpld3
    
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"

# GUIが使えない環境の場合はmatplotlibのバックエンドをnon-GUI modeに設定
import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    print("GUI環境が検出されなかったため、matplotlibのバックエンドをnon-GUI modeに設定")
    matplotlib.use("Agg")     


def population_rate_plot(populationratemonitors, time_end:int, time_start:int=0, save_path:str=None, smooth_window:int=None):
    """
    与えられたポピュレーションレートモニターから発火率のプロットを描画します。
    リストで複数のモニターを渡すと、それらを1枚のウィンドウにプロットします。
    表示する際には後ろにplt.show()が必要です。
    グラフを保存するには，保存するパスを渡します。
    発火率を平滑化する場合はsmooth_windowに平滑化するウィンドウサイズを渡します。
    シミュレーション全体での平均発火率も計算して表示します。

    Args:
        populationratemonitors (list): ポピュレーションレートモニターのリスト
        time_start (int): プロットする時間の開始範囲(ms) 
        time_end (int): プロットする時間の終了範囲(ms)
        save_path (str): 保存するパスの指定（オプション）
        smooth_window (int): 移動平均のウィンドウサイズ（オプション）。指定すると発火率を平滑化
    """
    def natural_sort_key(s):
        import re
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
    
    def plot_poprate_time_series_by_column(rates_mc, times, smooth_window=None):
        """
        カラム別のpopulation rateの時系列推移をプロットする内部関数
        """
        title = "Population Rate by Column (Only Excitatory Neurons)"
        fig_columns = plt.figure(figsize=(14, 3.3 * len(rates_mc)))
        fig_columns.canvas.manager.set_window_title(title)
                
        avg_rate = {}
        for i, (key, rate) in enumerate(rates_mc.items()):
            # 時間軸を準備
            if isinstance(times, (int, float)):
                times = np.array([times])
            elif len(times) == 0:
                continue
            times = np.unique(times)
            times.sort()
            
            # サブプロットを作成
            ax = plt.subplot(len(rates_mc), 1, i+1)
            
            # スムージング処理
            if smooth_window is not None:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed_rates = np.convolve(rate, kernel, mode='same')
                line = ax.plot(times, smoothed_rates, label=f'Column {key} (Smoothed)', 
                              linewidth=2, color='black')[0]
                ax.plot(times, rate, alpha=0.3, label=f'Column {key} (Original)', 
                      color='black', linestyle='--')
            else:
                line = ax.plot(times, rate, label=f'Column {key}', 
                              linewidth=2, color='black')[0]
            
            ax.set_ylabel('Rate (Hz)')
            ax.set_title(f'Column {key}', fontweight='bold')
            ax.legend(loc='upper right')
            
            # グリッド線を追加して可読性を向上
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 背景色を少し明るくして見やすくする
            ax.set_facecolor('#f9f9f9')
            
            # 最後のサブプロットにx軸ラベルを追加
            if i == len(rates_mc) - 1:
                ax.set_xlabel('Time (ms)')
            
            # カーソルを追加
            cursor = mplcursors.cursor(line, hover=True)
            
            @cursor.connect("add")
            def on_add(sel, key=key):
                sel.annotation.set_text(f'Column {key}\nTime: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz')
        
            # 全体の平均発火率を計算
            avg_rate[key] = np.mean(rate)
            avg_line = plt.axhline(y=avg_rate[key], color='k', linestyle='--', label=f'Avg: {avg_rate[key]:.2f}Hz', linewidth=1)
            plt.text(times[-1]*0.9, avg_rate[key]*1.05, f'{avg_rate[key]:.2f}Hz', 
                   fontsize=15, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7), color='black')

        # サブプロット間の間隔を調整
        plt.subplots_adjust(hspace=0.5)
        
        # 全体のタイトル
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.subplots_adjust(top=0.95)  # タイトル用のスペースを確保
        
        # グラフを保存
        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            column_save_path = os.path.normpath(os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_by_column{ext}"))
            plt.savefig(column_save_path, dpi=300)  # 解像度を300dpiに設定
            html_path = os.path.normpath(os.path.join(os.path.dirname(column_save_path), f"{os.path.basename(column_save_path).replace('.png', '')}.html"))
            mpld3.save_html(fig_columns, html_path)
            print(f"保存しました: {column_save_path}")
            print(f"保存しました: {html_path}")
        
        return fig_columns, avg_rate
    
    def plot_poprate_by_column_bar_chart(avg_rate):
        """
        カラムごとの平均発火率を棒グラフでプロットする内部関数
        """
        title = "Average Firing Rate (Column-wise) (Only Excitatory Neurons)"
        fig_bar = plt.figure(figsize=(10, 6))
        fig_bar.canvas.manager.set_window_title(title)
        
        column_names = list(avg_rate.keys())
        column_names.sort(key=natural_sort_key)  # 既存のnatural_sort_keyを使用
        rates = [avg_rate[name] for name in column_names]
        
        # 棒グラフの作成
        bars = plt.bar(range(len(column_names)), rates, color='black')
        
        # x軸のラベルを設定
        plt.xticks(range(len(column_names)), column_names)
        
        # 各バーの上に値を表示
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            plt.text(i, rate + 0.3, 
                    f'{rate:.2f}', ha='center', va='bottom', fontsize=12)
        
        plt.xlabel('Micro Circuit')
        plt.ylabel('Average Firing Rate (Hz)')
        plt.title(title)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # グラフを保存
        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            bar_save_path = os.path.normpath(os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_bar_chart{ext}"))
            plt.savefig(bar_save_path, dpi=300)  # 解像度を300dpiに設定
            html_path = os.path.normpath(os.path.join(os.path.dirname(bar_save_path), f"{os.path.basename(bar_save_path).replace('.png', '')}.html"))
            mpld3.save_html(fig_bar, html_path)
            print(f"保存しました: {bar_save_path}")
            print(f"保存しました: {html_path}")
        
        # plt.tight_layout()
        return fig_bar
    
    def plot_poprate_by_layer_heatmap(all_rates):
        """
        層ごとの発火率をヒートマップでプロットする内部関数
        """
        title = "Average Firing Rate by Layer (Column-wise) (Only Excitatory Neurons)"
        # カラム名を取得してソート
        columns = list(all_rates.keys())
        columns.sort(key=natural_sort_key)
        
        # 層の名前を取得（最初のカラムから）
        layers = list(all_rates[columns[0]].keys())
        # L23,L4,L5,L6の順番に並べる
        layers = ["L23", "L4", "L5", "L6"]
        
        # 層ごとのサブプロットを作成
        n_layers = len(layers)
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.manager.set_window_title(title)
        
        # ヒートマップデータの整形
        heatmap = np.zeros((len(layers), len(columns)))
        for i, layer in enumerate(layers):
            for j, col in enumerate(columns):
                if layer in all_rates[col]:
                    if "pyr" in all_rates[col][layer].keys():
                        heatmap[i, j] = np.mean(all_rates[col][layer]["pyr"])
                    elif "exc" in all_rates[col][layer].keys():
                        heatmap[i, j] = np.mean(all_rates[col][layer]["exc"])
                    else:
                        raise ValueError(f"Layer {layer} is not in all_rates[col][layer].keys()")
                    
        # ヒートマップのプロット
        ax = plt.subplot(111)
        im = ax.imshow(heatmap, cmap='viridis')
        
        # カラーバーを追加
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("Average Firing Rate (Hz)", rotation=270, labelpad=35, fontsize=25)
        
        # 各セルに値を表示
        for i in range(len(layers)):
            for j in range(len(columns)):
                text = ax.text(j, i, f'{heatmap[i, j]:.2f}',
                       ha="center", va="center", 
                       fontsize=25,
                       color="black" if heatmap[i, j] > np.max(heatmap)/2 else "white")
        
        # 軸ラベルの設定
        # ラベルのフォントサイズ調整
        ax.tick_params(axis='both', which='major')
        ax.set_xticks(np.arange(len(columns)))  # x軸の位置を設定
        ax.set_yticks(np.arange(len(layers)))   # y軸の位置を設定
        ax.set_xticklabels(columns)             # x軸のラベルを設定
        ax.set_yticklabels(layers)              # y軸のラベルを設定
        
        # 最後のサブプロットにx軸ラベルを追加
        ax.set_xlabel('Micro Circuit')
        ax.set_ylabel('Layer')
        
        # plt.tight_layout()
        plt.suptitle(title, fontsize=25)
        
        # グラフを保存
        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            layers_save_path = os.path.normpath(os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_layers_heatmap{ext}"))
            plt.savefig(layers_save_path, dpi=300)  # 解像度を300dpiに設定
            html_path = os.path.normpath(os.path.join(os.path.dirname(layers_save_path), f"{os.path.basename(layers_save_path).replace('.png', '')}.html"))
            mpld3.save_html(fig, html_path)
            print(f"保存しました: {layers_save_path}")
            print(f"保存しました: {html_path}")
        
        return fig
    
    def plot_popmon_time_series_by_allneurons(monitors, window_title, enabled_mc=False):
        """
        個々のポピュレーションをプロットする内部関数
        enabled_mc=Trueの場合、カラム別にウィンドウを分ける
        """
        if not enabled_mc:
            # 従来の動作：すべてのモニターを1つのウィンドウにプロット
            fig = plt.figure(figsize=(14, 3*len(monitors)))
            fig.canvas.manager.set_window_title(window_title)
            
            for i, mon in enumerate(monitors):
                ax = plt.subplot(len(monitors), 1, i+1)
                
                # データの取得と時間範囲の制限
                times = mon.t/ms
                rates = mon.rate/Hz
                mask = (times >= time_start) & (times <= time_end)
                times = times[mask]
                rates = rates[mask]
        
                # スムージング処理
                if smooth_window is not None:
                    kernel = np.ones(smooth_window) / smooth_window
                    smoothed_rates = np.convolve(rates, kernel, mode="same")
                    line_smoothed = plt.plot(times, smoothed_rates, label="Smoothed")[0]
                    line_original = plt.plot(times, rates, alpha=0.3, label="Original")[0]
                    plt.legend()
                    
                    # カーソルを追加（スムージング済みデータのみ）
                    cursor_smoothed = mplcursors.cursor(line_smoothed, hover=True)
                    
                    @cursor_smoothed.connect("add")
                    def on_add(sel):
                        sel.annotation.set_text(f"Smoothed\nTime: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz")
                else:
                    line = plt.plot(times, rates, color="black")[0]
                    cursor = mplcursors.cursor(line, hover=True)
                    
                    @cursor.connect("add")
                    def on_add(sel):
                        sel.annotation.set_text(f"Time: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz")
                
                # 平均発火率を計算して表示
                avg_rate = np.mean(rates)
                plt.axhline(y=avg_rate, color="black", linestyle="--", label=f"Avg: {avg_rate:.2f}Hz", linewidth=1)
                plt.text(times[-1]*0.9, avg_rate*1.05, f"{avg_rate:.2f}Hz", color="black", fontsize=15, fontweight="bold", bbox=dict(facecolor="white", alpha=0.7))
                
                # モニター名からレイヤー情報を抽出
                name_parts = mon.name.split("_")
                layer_name = next((part for part in name_parts if part.startswith("L")), "")
                plt.ylabel(f"Rate (Hz)")
                plt.title(f"{mon.name}")  # モニター名を表示
                    
                if i == len(monitors)-1:
                    plt.xlabel("Time (ms)")
                
                plt.grid(True)
            
            plt.suptitle(window_title, fontsize=20)
            plt.subplots_adjust(hspace=0.8)
            # plt.tight_layout()
            
            if save_path is not None:
                fig.savefig(save_path, dpi=300)  # 解像度を300dpiに設定
                print(f"保存しました: {save_path}")
            
            return fig
        
        else:
            # enabled_mc=Trueの場合、カラム別にグループ化してウィンドウを分ける
            # モニター名からカラム情報を抽出してグループ化
            column_groups = {}
            for mon in monitors:
                name_parts = mon.name.split("_")
                # カラム情報を抽出（例：M0, M1など）
                column_name = next((part for part in name_parts if part.startswith("M")), "Unknown")
                if column_name not in column_groups:
                    column_groups[column_name] = []
                column_groups[column_name].append(mon)
            
            figures = []
            
            # カラムごとに別々のウィンドウでプロット
            for col_name, monitors in column_groups.items():
                col_title = f"{window_title} - {col_name}"
                fig = plt.figure(figsize=(14, 3*len(monitors)))
                fig.canvas.manager.set_window_title(col_title)
                
                for i, mon in enumerate(monitors):
                    ax = plt.subplot(len(monitors), 1, i+1)
            
                    # データの取得と時間範囲の制限
                    times = mon.t/ms
                    rates = mon.rate/Hz
                    mask = (times >= time_start) & (times <= time_end)
                    times = times[mask]
                    rates = rates[mask]
            
                    # スムージング処理
                    if smooth_window is not None:
                        kernel = np.ones(smooth_window) / smooth_window
                        smoothed_rates = np.convolve(rates, kernel, mode="same")
                        line_smoothed = plt.plot(times, smoothed_rates, label="Smoothed")[0]
                        line_original = plt.plot(times, rates, alpha=0.3, label="Original")[0]
                        plt.legend()
                        
                        # カーソルを追加（スムージング済みデータのみ）
                        cursor_smoothed = mplcursors.cursor(line_smoothed, hover=True)
                        
                        @cursor_smoothed.connect("add")
                        def on_add(sel):
                            sel.annotation.set_text(f"Smoothed\nTime: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz")
                    else:
                        line = plt.plot(times, rates, color="black")[0]
                        cursor = mplcursors.cursor(line, hover=True)
                        
                        @cursor.connect("add")
                        def on_add(sel):
                            sel.annotation.set_text(f"Time: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz")
            
                    # 平均発火率を計算して表示
                    avg_rate = np.mean(rates)
                    plt.axhline(y=avg_rate, color="black", linestyle="--", label=f"Avg: {avg_rate:.2f}Hz", linewidth=1)
                    plt.text(times[-1]*0.9, avg_rate*1.05, f"{avg_rate:.2f}Hz", color="black", fontsize=15, fontweight="bold", bbox=dict(facecolor="white", alpha=0.7))
            
                    # モニター名からレイヤー情報を抽出
                    name_parts = mon.name.split("_")
                    layer_name = next((part for part in name_parts if part.startswith("L")), "")
                    plt.ylabel(f"Rate (Hz)")
                    plt.title(f"{mon.name}")  # モニター名を表示
                
                    if i == len(monitors)-1:
                        plt.xlabel("Time (ms)")
            
                    plt.grid(True)
            
                plt.suptitle(col_title, fontsize=20)
                plt.subplots_adjust(hspace=0.8)
                # plt.tight_layout()
                
                if save_path is not None:
                    base, ext = os.path.splitext(save_path)
                    current_save_path = os.path.normpath(os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_{col_name}{ext}"))
                    fig.savefig(current_save_path, dpi=300)  # 解像度を300dpiに設定
                    html_path = os.path.normpath(os.path.join(os.path.dirname(current_save_path), f"{os.path.basename(current_save_path).replace('.png', '')}.html"))
                    mpld3.save_html(fig, html_path)
                    print(f"保存しました: {current_save_path}")
                    print(f"保存しました: {html_path}")
                
                figures.append(fig)
            
            return figures
    
    # メイン関数の処理開始
    if not isinstance(populationratemonitors, list):
        populationratemonitors = [populationratemonitors]
        
    # time_endとtime_startから単位を削除
    if hasattr(time_end, 'dimensions'):
        time_end = float(time_end/ms)
    if hasattr(time_start, 'dimensions'):
        time_start = float(time_start/ms)
        
    # 全体のレート計算のためのデータを準備
    all_rates = {} # all_rates[mc_id][layer][neuron_type] = rates
    rates_mc = {}
    n_popmon_e = {} # 興奮性ニューロンのpopmonの数
    
    mc_enabled = False
    # データ整形
    for mon in populationratemonitors:
        times = mon.t/ms
        rates = mon.rate/Hz
        mask = (times >= time_start) & (times <= time_end)
        times = times[mask]
        rates = rates[mask]
        mc_id = mon.name.split('_')[0]
        if mc_id not in rates_mc.keys():
            rates_mc[mc_id] = np.zeros(len(times))
            n_popmon_e[mc_id] = 0
        if mc_id.startswith('M'):
            mc_enabled = True
            # ex: M1_popmon_L23_N_pyr
            neuron_type = mon.name.split('_')[-1]
            layer = mon.name.split('_')[2]
            assert neuron_type == "pyr" or neuron_type == "pv" or neuron_type == "vip" or neuron_type == "sst" or neuron_type == "inh" or neuron_type == "exc", "poppulation rate plot: neuron type is not correct!(mc_id: {}, neuron_type: {})".format(mc_id, neuron_type)
            assert layer == "L23" or layer == "L4" or layer == "L5" or layer == "L6", "poppulation rate plot: layer is not correct!(mc_id: {}, layer: {})".format(mc_id, layer)
            # mc_idと層ごとにpopデータを保存
            if "noise" not in mon.name.split("_"): # noiseニューロンのpoprateは保存しない
                if mc_id not in all_rates.keys():
                    all_rates[mc_id] = {}
                if layer not in all_rates[mc_id].keys():
                    all_rates[mc_id][layer] = {}
                if neuron_type not in all_rates[mc_id][layer].keys():
                    all_rates[mc_id][layer][neuron_type] = rates
                # mc_id毎にpyrニューロンのpoprateのデータを保存
                if neuron_type == "pyr" or neuron_type == "exc":
                    rates_mc[mc_id] += rates
                    n_popmon_e[mc_id] += 1
    
    figs = []
    if mc_enabled:
        for key in rates_mc.keys():
            rates_mc[key] = rates_mc[key]/n_popmon_e[key]
        print(rates_mc)
        
        # カラム別population rateの推移プロット
        fig_columns, avg_rate = plot_poprate_time_series_by_column(rates_mc, times, smooth_window)
        figs.append(fig_columns)
        
        # カラムごとの平均発火率を棒グラフでプロット
        if avg_rate:
            fig_bar = plot_poprate_by_column_bar_chart(avg_rate)
            figs.append(fig_bar)
        
        # 層別の発火率ヒートマップ
        fig_heatmap = plot_poprate_by_layer_heatmap(all_rates)
        figs.append(fig_heatmap)
    
    window_title = "Population Rate Plot"
    fig = plot_popmon_time_series_by_allneurons(populationratemonitors, window_title, enabled_mc=mc_enabled)
    if isinstance(fig, list):
        figs.extend(fig)
    else:
        figs.append(fig)
        
    return figs

    
def raster_plot(spikemons, time_end:int, time_start:int=0, save_path:str=None):
    """
    与えられたスパイクモニターからラスタプロットを描画します。
    リストで複数のスパイクモニターを渡すと、それらを1枚のウィンドウにプロットします。
    モニター名が'M'で始まる場合は、カラム別に分割して表示します。
    表示する際には後ろにplt.show()が必要です。
    グラフを保存するには，保存するパスを渡します。
    グラフは，pngとhtmlの2種類で保存されます。

    Args:
        spikemons (list of SpikeMonitor): スパイクモニターのリスト
        time_start (int): プロットする時間の開始範囲(ms)
        time_end (int): プロットする時間の終了範囲(ms)
        save_path (str): 保存するパスの指定（オプション）
    """
    window_title = "Raster plot"
    if not isinstance(spikemons, list):
        spikemons = [spikemons]
    # time_endとtime_startから単位を削除
    if hasattr(time_end, 'dimensions'):
        time_end = float(time_end/ms)
    if hasattr(time_start, 'dimensions'):
        time_start = float(time_start/ms)
        
    # マイクロカラム（M0, M1など）が存在するか確認
    mc_enabled = False
    for mon in spikemons:
        if mon.name.split("_")[0].startswith("M"):
            mc_enabled = True
            break
    
    figs = []
    
    if mc_enabled:
        # カラム別にモニターをグループ化
        column_groups = {}
        for mon in spikemons:
            column_name = mon.name.split("_")[0]
            if column_name not in column_groups:
                column_groups[column_name] = []
            column_groups[column_name].append(mon)
        
        # カラムごとに別々のウィンドウでプロット
        for col_name, monitors in column_groups.items():
            col_title = f"Raster plot - {col_name}"
            
            fig = plt.figure(figsize=(14, 2*len(monitors)))
            fig.canvas.manager.set_window_title(col_title)
            plt.suptitle(col_title)
            
            # サブプロットを作成
            axes = []
            scatter_plots = []
            for this_row in range(len(monitors)):
                if this_row == 0:
                    ax = plt.subplot(len(monitors), 1, this_row+1)
                else:
                    ax = plt.subplot(len(monitors), 1, this_row+1, sharex=axes[0])
                axes.append(ax)
                
                scatter = ax.scatter(monitors[this_row].t/ms, monitors[this_row].i, 
                                   s=1, c='k', marker='.')
                scatter_plots.append(scatter)
                
                if this_row+1 == len(monitors):
                    ax.set_xlabel('Time (ms)')
                ax.set_xlim(time_start, time_end)
                ax.set_ylim(-1, len(monitors[this_row].source))
                ax.set_ylabel('Neuron index')
                if hasattr(monitors[this_row], 'name'):
                    ax.set_title(monitors[this_row].name)
                else:
                    ax.set_title(f"Monitor {this_row}")
                    
                plt.subplots_adjust(hspace=1)
                    
                if save_path is not None:
                    base, ext = os.path.splitext(save_path)
                    current_save_path = os.path.normpath(os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_{col_name}{ext}"))
                    plt.savefig(current_save_path)
                    html_path = os.path.normpath(os.path.join(os.path.dirname(current_save_path), f"{os.path.basename(current_save_path).replace('.png', '')}.html"))
                    mpld3.save_html(fig, html_path)
                    print(f"保存しました: {current_save_path}")
                    print(f"保存しました: {html_path}")
            
            # 各スキャッタープロットにカーソルを追加
            for scatter in scatter_plots:
                cursor = mplcursors.cursor(scatter, hover=False)
                
                @cursor.connect("add")
                def on_add(sel):
                    neuron_idx = int(sel.target[1])
                    time = sel.target[0]
                    sel.annotation.set_text(f'Neuron: {neuron_idx}\nTime: {time:.2f}ms')
                    sel.annotation.xy = (sel.target[0], sel.target[1])
            
            plt.subplots_adjust(hspace=1.0)
            # plt.tight_layout()

    else:
        # 従来の動作：すべてのモニターを1つのウィンドウに表示
        fig = plt.figure(figsize=(14, 2*len(spikemons)))
        window_title = "Raster plot"
    
    plt.suptitle(window_title)
    try:
        fig.canvas.manager.set_window_title(window_title)
    except AttributeError:
        # 非対話型バックエンドの場合は処理をスキップ
        pass
    
    # サブプロットを作成
    axes = []
    scatter_plots = []
    for this_row in range(len(spikemons)):
        if this_row == 0:
            ax = plt.subplot(len(spikemons), 1, this_row+1)
        else:
            ax = plt.subplot(len(spikemons), 1, this_row+1, sharex=axes[0])
        axes.append(ax)
        
        scatter = ax.scatter(spikemons[this_row].t/ms, spikemons[this_row].i, 
                           s=1, c='k', marker='.')
        scatter_plots.append(scatter)
        
        if this_row+1 == len(spikemons):
            ax.set_xlabel('Time (ms)')
        ax.set_xlim(time_start, time_end)
        ax.set_ylim(-1, len(spikemons[this_row].source))
        ax.set_ylabel('Neuron index')
        if hasattr(spikemons[this_row], 'name'):
            ax.set_title(spikemons[this_row].name)
        else:
            ax.set_title(f"Monitor {this_row}")
                
            if save_path is not None:
                try:
                    plt.savefig(os.path.normpath(save_path), dpi=300)  # 解像度を300dpiに設定
                    html_path = os.path.normpath(os.path.join(os.path.dirname(save_path), f"{os.path.basename(save_path).replace('.png', '')}.html"))
                    mpld3.save_html(fig, html_path)
                    print(f"保存しました: {save_path}")
                    print(f"保存しました: {html_path}")
                except Exception as e:
                    print(f"ファイル保存中にエラーが発生しました: {e}")
    
    # 各スキャッタープロットにカーソルを追加
    for scatter in scatter_plots:
        cursor = mplcursors.cursor(scatter, hover=False)
        
        @cursor.connect("add")
        def on_add(sel):
            neuron_idx = int(sel.target[1])
            time = sel.target[0]
            sel.annotation.set_text(f'Neuron: {neuron_idx}\nTime: {time:.2f}ms')
            sel.annotation.xy = (sel.target[0], sel.target[1])
    
        plt.subplots_adjust(hspace=1.0)
        # plt.tight_layout()
    
    figs.append(fig)
    
    return figs

def state_plot(statemon:StateMonitor|list, time_end:int, time_start:int=0, neuron_num:int=0, variable_names:list=None, save_path:str=None):
    """
    与えられたステートモニターからプロットを描画します。この関数実行後にplt.show()などを記述する必要があります。
    変数のリストを渡すと，すべての変数のプロットを縦に並べて同に描画します。
    変数のリストを渡さない場合は，記録されているすべての変数をプロットします。
    IeとIi、geとgiがある場合は重ねて描画します。
    グラフを保存するには，保存するパスを渡します。
    
    Args:
        statemon (StateMonitor|list): ステートモニターまたはステートモニターのリスト
        neuron_num (int): プロットするニューロンの番号
        variable_names (list): プロットする変数の名前。Noneの場合は記録されているすべての変数をプロット
        fig_title (str): フィグのタイトル
        time_start (int): プロットする時間の開始範囲(ms)
        time_end (int): プロットする時間の終了範囲(ms)
    """        
    
    # time_endとtime_startから単位を削除し、確実にms単位に統一
    if hasattr(time_end, 'dimensions'):
        time_end = float(time_end/ms)  # ms単位に変換
    if hasattr(time_start, 'dimensions'):
        time_start = float(time_start/ms)  # ms単位に変換
    
    # time_startとtime_endは常にミリ秒単位とする
    time_start_ms = time_start
    time_end_ms = time_end
    
    # statemonがリストでない場合はリストに変換
    if not isinstance(statemon, list):
        statemon = [statemon]
    
    figs = []
    for monitor in statemon:
        # variable_namesがNoneの場合は記録されているすべての変数を使用
        if variable_names is None:
            monitor_vars = list(monitor.record_variables)
        else:
            monitor_vars = variable_names
        
        # IeとIi、geとgiのペアをチェック
        has_Ie = "Ie" in monitor_vars
        has_Ii = "Ii" in monitor_vars
        has_ge = "ge" in monitor_vars
        has_gi = "gi" in monitor_vars
        
        # プロットする変数リストを整理
        plot_vars = []
        if has_Ie:
            plot_vars.append(("Ie", ["Ie"]))
        if has_Ii:
            plot_vars.append(("Ii", ["Ii"]))
        if has_Ie or has_Ii:
            plot_vars.append(("I", ["Ie", "Ii"]))
            if has_Ie and has_Ii:  # 両方存在する場合は和も追加
                plot_vars.append(("I (E-I)", ["Ie-Ii"]))
        if has_ge or has_gi:
            plot_vars.append(("g", ["ge", "gi"]))
        for var in monitor_vars:
            if var not in ["Ie", "Ii", "ge", "gi"]:
                plot_vars.append((var, [var]))
                
        fig = plt.figure(figsize=(14, 3.0*len(plot_vars)))
        plt.suptitle(f"State plot - {monitor.name} - Neuron {neuron_num}")
        fig.canvas.manager.set_window_title(f"State plot - {monitor.name} - Neuron {neuron_num}")        
        # サブプロットを作成
        axes = []
        line_plots = []
        for this_row, (plot_name, vars_to_plot) in enumerate(plot_vars):
            if this_row == 0:
                ax = plt.subplot(len(plot_vars), 1, this_row+1)
            else:
                ax = plt.subplot(len(plot_vars), 1, this_row+1, sharex=axes[0])
            axes.append(ax)
            
            # 軸の範囲を固定するために、データの最小値と最大値を取得
            y_min = float('inf')
            y_max = float('-inf')
            has_valid_data = False

            # データの単位を取得・調整するための変数
            y_data_unit = None
            y_data_values = []

            for var in vars_to_plot:
                if var == "Ie-Ii" and "Ie" in monitor_vars and "Ii" in monitor_vars:
                    data = getattr(monitor, "Ie")[neuron_num] - getattr(monitor, "Ii")[neuron_num]
                    if len(data) > 0:
                        # 単位情報の取得方法を改善
                        if hasattr(data, 'get_best_unit'):
                            # Brian2のQuantityオブジェクトの場合
                            y_data_unit = data.get_best_unit()
                            data_scaled = data/y_data_unit
                        else:
                            # numpyの場合はそのまま使用
                            y_data_unit = 1  # スケーリングなし
                            data_scaled = data
                        y_min = min(y_min, np.min(data_scaled))
                        y_max = max(y_max, np.max(data_scaled))
                        

                        # その他の場合（numpy配列など）
                        try:
                            t_axis = monitor.t/ms
                        except:
                            t_axis = monitor.t
                        
                        line = ax.plot(t_axis, data_scaled, color="g", label="E-I")[0]
                        ax.axhline(y=0, color='gray', linestyle='--', alpha=1)
                        line_plots.append((line, var))
                        has_valid_data = True
                        y_data_values.extend(data_scaled)
                elif var in monitor_vars:
                    try:
                        data = getattr(monitor, var)[neuron_num]
                        if len(data) > 0:
                            # 単位情報の取得方法を改善
                            if y_data_unit is None:
                                if hasattr(data, 'get_best_unit'):
                                    # Brian2のQuantityオブジェクトの場合
                                    y_data_unit = data.get_best_unit()
                                elif hasattr(monitor, 'get_best_unit'):
                                    # StateMonitorDataオブジェクトの場合
                                    y_data_unit = monitor.get_best_unit(var)
                                else:
                                    # 単位情報がない場合
                                    y_data_unit = 1  # スケーリングなし
                            
                            # データのスケーリング
                            try:
                                data_scaled = data/y_data_unit
                            except:
                                # numpyの場合はそのまま
                                data_scaled = data
                            
                            y_min = min(y_min, np.min(data_scaled))
                            y_max = max(y_max, np.max(data_scaled))
                            
                            if var in ["ge", "Ie"]:
                                color = "r"
                                label = "Excitatory"
                            elif var in ["gi", "Ii"]:
                                color = "b"
                                label = "Inhibitory"
                            else:
                                color = "k"
                                label = var
                            
                            # 時間軸の単位処理
                            try:
                                t_axis = monitor.t/ms
                            except:
                                t_axis = monitor.t
                            
                            line = ax.plot(t_axis, data_scaled, color=color, label=label)[0]
                            line_plots.append((line, var))
                            has_valid_data = True
                            y_data_values.extend(data_scaled)
                    except (AttributeError, IndexError):
                        continue
            
            # データが存在する場合のみ軸の範囲とラベルを設定
            if has_valid_data:
                margin = (y_max - y_min) * 0.1
                if abs(y_max - y_min) < 1e-10:  # 値がほぼ同じ場合
                    y_min -= 0.1
                    y_max += 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
                
                # 単位を含む軸ラベルを設定
                if hasattr(y_data_unit, '__str__'):
                    unit_str = str(y_data_unit)
                else:
                    unit_str = str(y_data_unit)
                
                # 単位表示の改善
                if unit_str == '1':
                    ax.set_ylabel(plot_name)
                else:
                    # StateMonitorDataオブジェクトの場合、単位が文字列
                    ax.set_ylabel(f'{plot_name} ({unit_str})')
            else:
                # データが存在しない場合はデフォルトの範囲を設定
                ax.set_ylim(-1, 1)
                ax.text(0.5, 0.5, 'No data available', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)

            if len(vars_to_plot) > 1:
                # ラベル付きの線があるか確認してから凡例を表示
                has_labeled_artists = False
                for artist in ax.get_children():
                    if hasattr(artist, 'get_label') and not artist.get_label().startswith('_'):
                        has_labeled_artists = True
                        break
                
                if has_labeled_artists:
                    ax.legend()
                
            if this_row+1 == len(plot_vars):
                ax.set_xlabel('Time (ms)')
            
            # 適切な時間範囲を設定
            ax.set_xlim(time_start_ms, time_end_ms)
        
        # 各線プロットにカーソルを追加
        for line, var_name in line_plots:
            cursor = mplcursors.cursor(line, hover=False)
            
            @cursor.connect("add")
            def on_add(sel, var_name=var_name):
                time = sel.target[0]
                value = sel.target[1]
                sel.annotation.set_text(f'{var_name}\nTime: {time:.2f}ms\nValue: {value:.4f}')
                sel.annotation.xy = (sel.target[0], sel.target[1])
                
                # アノテーションの自動ズームを無効化
                ax = sel.artist.axes
                ax.autoscale(enable=False)
        
        plt.subplots_adjust(hspace=1.0)
        # plt.tight_layout()
        if save_path is not None:
            # 複数のモニターがある場合はファイル名に番号を付加
            if len(statemon) > 1:
                base, ext = os.path.splitext(save_path)
                monitor_save_path = os.path.normpath(os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_{monitor.name}{ext}"))
            else:
                monitor_save_path = os.path.normpath(save_path)
            plt.savefig(monitor_save_path, dpi=300)  # 解像度を300dpiに設定
            html_path = os.path.normpath(os.path.join(os.path.dirname(monitor_save_path), f"{os.path.basename(monitor_save_path).replace('.png', '')}.html"))
            mpld3.save_html(fig, html_path)
            print(f"保存しました: {monitor_save_path}")
            print(f"保存しました: {html_path}")
        figs.append(fig)
    
    return figs
    
def weight_plot_1_neuron(synapse, neuron_idx, n_pre, n_post):
    """
    与えられたシナプスグループから、指定されたニューロンの重みのマップを描画します。

    Args:
        synapse (SynapseGroup): シナプスグループ
        neuron_idx (int): ニューロンのインデックス
    """
    weight_mat = np.zeros((n_pre, n_post))
    for i, j, w in zip(synapse.i, synapse.j, synapse.w):
        weight_mat[i, j] = w
    
    weight_mat_plot = weight_mat[:, neuron_idx].reshape(int(np.sqrt(n_pre)), int(np.sqrt(n_pre)))
    plt.imshow(weight_mat_plot, cmap="viridis")

            
def weight_plot(synapse, n_pre, n_post, title="", save_fig=False, save_path:str=None, n_this_fig=0, assigned_labels=None):
    """
    与えられたステートモニターから重みのプロットを描画します。

    Args:
        synapse (SynapseGroup or list or np.ndarray): シナプスグループか重みの行列
        n_pre (int): 前のニューロンの数
        n_post (int): 後のニューロンの数
        save_fig (bool): フィグを保存するかどうか
        save_path (str): フィグを保存するパス
        n_this_fig (int): このフィグの番号（保存するのファイル名にな)
        assigned_labels (list): 割り当てられたラベルのリスト
    """
    # synapse.w[neuron_idx][time_idx]
    weight_mat = np.zeros((n_pre, n_post))
    if isinstance(synapse, Synapses):
        for i, j, w in zip(synapse.i, synapse.j, synapse.w):
            weight_mat[i, j] = w
    elif isinstance(synapse, list) or isinstance(synapse, np.ndarray):
        weight_mat = np.array(synapse)
        weight_mat = weight_mat.reshape(n_pre, n_post)
    else:
        raise ValueError("synapseはSynapsesクラスのインスタンスか，重みの行列である必要があります。")
        
    # サブプロットの行数と列数を計算
    n_rows = int(np.ceil(np.sqrt(n_post)))
    n_cols = int(np.ceil(n_post / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 12))  # 幅を10に増やして右側に空白を作る
    axes = axes.flatten()  # 2次元配列を1次元に変換

    for img in range(n_post):
        weightloop = weight_mat[:, img].reshape(
            int(np.sqrt(n_pre)), int(np.sqrt(n_pre))
        )
        cax = axes[img].matshow(weightloop, cmap="viridis")
        axes[img].set_xticks([])
        axes[img].set_yticks([])
        
        if assigned_labels is not None:
            axes[img].set_xlabel(f"Neuron {img},label:{assigned_labels[img]}", fontsize=6)
        else:
            axes[img].set_xlabel(f"Neuron {img}", fontsize=6)

    # 余分なサブプロットを非表にする
    for img in range(n_post, n_rows * n_cols):
        axes[img].axis('off')

    fig.suptitle(f"Weight plot - {synapse.name}")
    fig.canvas.manager.set_window_title(f"Weight plot - {synapse.name}")

    # 右側にカラーバーを配置
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 右側に空白を作り、そこにカラーバーを配置
    plt.colorbar(cax, cax=cbar_ax)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) # カラーバーの警告を無視
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 右側に10%の空白を確保
    warnings.filterwarnings("default", category=UserWarning) # カラーバーの警告を有効化
    
    if save_path is not None:
        if n_this_fig == 0:
            # save_path内の全ファイルを削除
            files = glob.glob(os.path.join(save_path, '*.png'))
            for f in files:
                os.remove(f)
                print(f"\tDeleted {f}")
        save_file_path = os.path.normpath(os.path.join(save_path, f"{n_this_fig}.png"))
        plt.savefig(save_file_path, dpi=300)  # 解像度を300dpiに設定
        print(f"保存しました: {save_file_path}")
        
        plt.clf()
        plt.close()
    
def firing_rate_heatmap(spikemon, start_time, end_time, save_fig=False, save_path:str=None, n_this_fig=None):
    """
    与えられたスパイクモニターから発火率のヒートマップを描画します。

    Args:
        spikemon (SpikeMonitor): スパイクモニター
        start_time (float): 開始時間
        end_time (float): 終了時間
        save_fig (bool): フィグを保存するかどうか
        save_path (str): フィグを保存するパス
        n_this_fig (int): のフィグの番号（保存する際のファイル名になる)
    Returns:
        heatmap_data (np.ndarray): 発火率のヒートマップのデータ
    """

    firing_rates = tools.get_firing_rate(spikemon, start_time, end_time, mode="count")
    n_neurons = len(firing_rates)
    
    # 最適な行数と列数を計算
    n_rows = int(np.ceil(np.sqrt(n_neurons)))
    n_cols = int(np.ceil(n_neurons / n_rows))
    
    heatmap_data = np.full((n_rows, n_cols), np.nan)
    for i in range(n_neurons):
        row = i // n_cols
        col = i % n_cols
        heatmap_data[row, col] = firing_rates[i]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f', cbar=True, mask=np.isnan(heatmap_data), vmin=0)
    plt.title(f'Firing Count Heatmap ({start_time} to {end_time})')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{n_this_fig}_count_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"保存しました: {os.path.join(save_path, f'{n_this_fig}_count_heatmap.png')}")
    else:
        plt.show()
    
    plt.clf()
    plt.close()
    
    return heatmap_data

def visualize_wta_response(input_image, synapse, spikemon, start_time:int, exposure_time:int, save_path=None, n_this_fig=None):
    """
    Parameters:
    -----------
    input_image : np.array (height, width)
        入力画像
    synapse : SynapseGroup
        シナプスグループ
    spikemon : SpikeMonitor
        スパイクモニター
    start_time : int
        開始時間
    exposure_time : int
        露光時間
    """
    n_i = synapse.N_incoming_post[0]
    n_j = synapse.N_outgoing_pre[0]
    
    weight_mat = np.zeros((n_i, n_j))
    for i, j, w in zip(synapse.i, synapse.j, synapse.w):
        weight_mat[i, j] = w
    # VariableViewをNumPy配列に変換
    weights = np.array(weight_mat)

    # スパイクカウントの計算
    n_neurons = n_j
    spike_counts = np.zeros(n_neurons)
    # start_timeからexposure_time経過後までのスパイクのみを集計
    end_time = start_time + exposure_time
    spikes = tools.get_spikes_within_time_range(spikemon=spikemon, start_time=start_time, end_time=end_time)
    spike_indices = [spike[1] for spike in spikes]
    unique, counts = np.unique(spike_indices, return_counts=True)
    spike_counts[unique] = counts
    
    # 上位5位までのニューロンを特定
    top_5_winners = np.argsort(spike_counts)[::-1][:5] if len(spike_counts) > 0 else [0] * 5
    
    # プロットのサイズを調整
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(5, 5, figure=fig)
    
    # 1. 入力画像 (左上)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(input_image, cmap='gray')
    ax_input.set_title('Input Image')
    ax_input.axis('off')
    
    # 2. 上位5位のニューロンの重み (右側)
    for i, winner in enumerate(top_5_winners):
        ax = fig.add_subplot(gs[i, 1])
        winner_weights = weights[:, winner].reshape((int(np.sqrt(n_i)), int(np.sqrt(n_i))))
        ax.imshow(winner_weights, cmap='gray')
        ax.set_title(f'#{i+1} Neuron(#{winner})\nSpike Count: {spike_counts[winner]:.0f}')
        ax.axis('off')
    
    # 3. 1位のニューロンの重みと入力の重なり (左中央)
    for i, winner in enumerate(top_5_winners): 
        ax = fig.add_subplot(gs[i, 2])
        winner_weights = weights[:, winner].reshape((int(np.sqrt(n_i)), int(np.sqrt(n_i))))
        overlap = input_image * winner_weights
        ax.imshow(overlap, cmap='coolwarm')
        ax.set_title(f'#{i+1} Neuron(#{winner})\nOverlap of Input and Weight\n(Red: Positive, Blue: Negative)')
        ax.axis('off')
    
    # 4. スパイク発火頻度 (左下)
    ax_spikes = fig.add_subplot(gs[1:4, 3:5])
    bar_colors = ['gray'] * n_neurons  # デフォルトは全てグレー

    # 上位5位のニューロンの色を設定
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    for winner, color in zip(top_5_winners, colors):
        bar_colors[winner] = color

    # 色分けされた棒グラフを描画
    bars = ax_spikes.bar(range(n_neurons), spike_counts, color=bar_colors)
    ax_spikes.set_title('Spike Count per Neuron')
    ax_spikes.set_xlabel('Neuron ID')
    ax_spikes.set_xticks(np.arange(0, n_neurons+1, 10))
    ax_spikes.set_ylabel('Spike Count')

    # 凡例を追加
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=f'#{i+1} Winner (Neuron #{winner})')
                        for i, (winner, color) in enumerate(zip(top_5_winners, colors))]
    ax_spikes.legend(handles=legend_elements)

    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'wta_response_{n_this_fig}.png'), dpi=300)  # 解像度を300dpiに設定
        print(f"保存しました: {os.path.join(save_path, f'wta_response_{n_this_fig}.png')}")
        plt.close()
    
    return fig

def plot_all_monitors(network, time_end:int=None, time_start:int=0, save_dir_path:str=None, smooth_window:int=None, monitor_type:list=None):
    """
    ネットワーク内のすべてのスパイクモニター、ステートモニター、ポピュレーションレートモニターを描画します。
    spikemon, statemon, popmonのディレクトリを作成し，その中にそれぞれのモニターを保存します。

    Args:
        network (dict or Network): ネットワークの辞書またはNetworkオブジェクト
        time_end (int): プロットする時間の終了範囲(ms)
        time_start (int): プロットする時間の開始範囲(ms)
        save_dir_path (str): 保存するパスの指定（オプション）
        smooth_window (int): ポピュレーションレートモニターの平滑化ウィンドウサイズ（オプション）
        monitor_type (list): 描画するモニターのタイプのリスト。"popmon", "spikemon", "statemon", "all"のいずれかを含む。
                           デフォルトは["all"]で、すべてのタイプのモニターを描画します。
    """
    # monitor_typeのデフォルト値を設定
    if monitor_type is None:
        monitor_type = ["all"]
    
    # 文字列で渡された場合はリストに変換
    if isinstance(monitor_type, str):
        monitor_type = [monitor_type]
        
    # make dirs
    if "spikemon" in monitor_type or "all" in monitor_type:
        os.makedirs(os.path.join(save_dir_path, "spikemon"), exist_ok=True)
    if "statemon" in monitor_type or "all" in monitor_type:
        os.makedirs(os.path.join(save_dir_path, "statemon"), exist_ok=True)
    if "popmon" in monitor_type or "all" in monitor_type:
        os.makedirs(os.path.join(save_dir_path, "popmon"), exist_ok=True)
    
    # 各種モニターを分類
    spike_monitors = []
    state_monitors = []
    population_rate_monitors = []
    if time_end is None:
        time_end = network.t
    
    # ネットワークがdictの場合とNetworkオブジェクトの場合で処理を分ける
    if hasattr(network, 'items'):  # dictの場合
        items = network.items()
    else:  # Networkオブジェクトの場合
        items = [(obj.name, obj) for obj in network.objects if hasattr(obj, 'name')]
    
    for key, value in items:
        if isinstance(key, str):  # キーが文字列の場合
            if "spikemon" in key:
                spike_monitors.append(value)
            elif "statemon" in key:
                state_monitors.append(value)
            elif "popmon" in key:
                population_rate_monitors.append(value)
        else:  # キーが文字列でない場合は、オブジェクトの型で判断
            from brian2.monitors import SpikeMonitor, StateMonitor, PopulationRateMonitor
            if isinstance(value, SpikeMonitor):
                spike_monitors.append(value)
            elif isinstance(value, StateMonitor):
                state_monitors.append(value)
            elif isinstance(value, PopulationRateMonitor):
                population_rate_monitors.append(value)
    
    figs = []
    
    # monitor_typeリストに基づいて描画するモニターを選択
    if "all" in monitor_type or "spikemon" in monitor_type:
        # スパイクモニターをプロット（レイヤー別に分ける処理を削除）
        if spike_monitors:
            if save_dir_path is not None:
                spikemon_save_path = os.path.join(save_dir_path, 'spikemon', 'raster_plot.png')
                print(f"保存パス: {spikemon_save_path}")
            else:
                spikemon_save_path = None
            figs.extend(raster_plot(spike_monitors, time_end=time_end, time_start=time_start, save_path=spikemon_save_path))
    
    # ステートモニターをプロット
    if ("all" in monitor_type or "statemon" in monitor_type) and state_monitors:
        if save_dir_path is not None:
            state_save_path = os.path.join(save_dir_path, 'statemon', 'state_plot.png')
            print(f"保存パス: {state_save_path}")
        else:
            state_save_path = None
        figs.extend(state_plot(state_monitors, time_end=time_end, time_start=time_start, save_path=state_save_path))

    # ポピュレーションレートモニターをプロット
    if ("all" in monitor_type or "popmon" in monitor_type) and population_rate_monitors:
        if save_dir_path is not None:
            poprate_save_path = os.path.join(save_dir_path, 'popmon', 'population_rate_plot.png')
            print(f"保存パス: {poprate_save_path}")
        else:
            poprate_save_path = None
        figs.extend(population_rate_plot(population_rate_monitors, time_end=time_end, time_start=time_start, save_path=poprate_save_path, smooth_window=smooth_window))
    
    return figs