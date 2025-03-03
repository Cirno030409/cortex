from brian2 import *
from Brian2_Framework.Tools import *
from Brian2_Framework.Plotters import *
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from Brian2_Framework.Tools import SpikeMonitorData
from Brian2_Framework.Tools import StateMonitorData
import os
import time
import json
import matplotlib.lines
import numpy as np

file_path = []

# グローバル変数としてすべてのアクティブなfigureを追跡
all_active_figures = set()

# グローバル変数として画像ラベルを保持
image_labels = None

def handle_drop(event):
    global file_path
    global image_labels
    # 複数ファイルのパスを取得
    if event.data.startswith('{'):
        files = event.data.split('} {')
        files = [f.strip('{}') for f in files]
    else:
        files = event.data.split(' ')
    
    # ファイルの拡張子で処理を分岐
    pkl_files = [f for f in files if f.endswith('.pkl')]
    json_files = [f for f in files if f.endswith('.json')]
    
    if pkl_files:
        plot_monitor(pkl_files)
    
    if json_files:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    image_labels = json.load(f)
                # 設定ウィンドウが開いている場合、ラベルリストを更新
                for window in root.winfo_children():
                    if isinstance(window, tk.Toplevel) and window.title() == "Plot Settings":
                        update_labels_in_window(window, image_labels)
            except Exception as e:
                tk.messagebox.showerror("Error", f"JSONファイルの読み込みに失敗しました: {str(e)}")

def update_labels_in_window(window, labels):
    # 既存のラベルリストボックスを探す
    for widget in window.winfo_children():
        if isinstance(widget, tk.LabelFrame) and widget.cget("text") == "Loaded Files":
            listbox = None
            for child in widget.winfo_children():
                if isinstance(child, tk.Frame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, tk.Listbox):
                            listbox = grandchild
                            break
            
            if listbox:
                # 現在の選択状態を保存
                selected_indices = listbox.curselection()
                
                # 各項目にラベルを追加（存在する場合）
                for i in range(listbox.size()):
                    current_text = listbox.get(i)
                    file_name = current_text.split(']')[0][1:]  # [filename] の形式から filename を抽出
                    
                    if file_name in labels:
                        new_text = f"{current_text} - {labels[file_name]}"
                        listbox.delete(i)
                        listbox.insert(i, new_text)
                
                # 選択状態を復元
                for index in selected_indices:
                    listbox.selection_set(index)

def plot_monitor(file_paths):
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    # 複数のモニターデータを格納するリスト
    spike_monitors = []
    state_monitors = []
    
    # ファイルの分類
    for file_path in file_paths:
        monitor = tools.load_monitor(file_path)
        if isinstance(monitor, SpikeMonitorData):
            spike_monitors.append(monitor)
        elif isinstance(monitor, StateMonitorData):
            state_monitors.append(monitor)
    
    # 全てのモニターから最大時間を取得
    max_simulation_time = 0
    for file_path in file_paths:
        monitor = tools.load_monitor(file_path)
        if len(monitor.t) == 0:
            tk.messagebox.showerror("Error", f"モニターファイル {file_path} にデータがありません。")
            continue
        if isinstance(monitor, (SpikeMonitorData, StateMonitorData)):
            max_simulation_time = max(max_simulation_time, float(monitor.t[-1]*1000))
    
    # プロットウィンドウの作成
    plot_window = tk.Toplevel(root)
    plot_window.title("Plot Settings")
    plot_window.geometry("700x500")
    
    # 各plot_settingsウィンドウ固有のfiguresリストを作成
    plot_window.figures = []
    plot_window.active_figures = set()
    
    def on_figure_closed(event):
        # グローバルとローカル両方のセットから削除
        if event.canvas.figure in plot_window.active_figures:
            plot_window.active_figures.remove(event.canvas.figure)
        if event.canvas.figure in all_active_figures:
            all_active_figures.remove(event.canvas.figure)
    
    # 時間設定用スライダー - 既存のプロットがある場合は表示しない
    time_slider = None
    if not all_active_figures:
        time_frame = tk.LabelFrame(plot_window, text="Time Window (ms)")
        time_frame.pack(fill='x', padx=10, pady=5)
        time_slider = tk.Scale(time_frame, from_=0, to=max_simulation_time, orient=tk.HORIZONTAL)
        time_slider.set(150)
        time_slider.pack(fill='x', padx=5)
    
    def plot_all():
        global fig_spike, fig_state, on_spike_click, on_state_click
        
        # 時間範囲の設定
        if time_slider:
            time_window = time_slider.get()
        else:
            # 既存のプロットから時間範囲を取得
            first_fig = next(iter(all_active_figures))
            time_window = first_fig.get_axes()[0].get_xlim()[1]
        
        # 既存のプロットはそのままに、新しいプロットを追加
        new_figures = []
        
        # 選択されているインデックスを取得
        selected_indices = monitor_listbox.curselection()
        selected_monitors = [all_monitors[i] for i in selected_indices]
        
        # Spikeモニターとステートモニターを分離
        selected_spike_monitors = [monitor for type_, monitor in selected_monitors if type_ == 'spike']
        selected_state_monitors = [monitor for type_, monitor in selected_monitors if type_ == 'state']
        
        if selected_spike_monitors:
            # raster_plot関数がリストを返しているため、最初の要素を取得
            fig_list = raster_plot(selected_spike_monitors, time_end=time_window)
            fig_spike = fig_list[0] if isinstance(fig_list, list) else fig_list
            
            # スパイクプロットにクリックイベントを追加
            def on_spike_click(event):
                # 他のイベントハンドラからの呼び出しを防ぐ
                if not hasattr(event, 'spike_click_processed'):
                    event.spike_click_processed = True
                else:
                    return
                
                if event.inaxes and event.button == 1:  # 左クリックの場合
                    x, y = event.xdata, event.ydata
                    if x is not None and y is not None:
                        # 全てのfigureの全てのaxesから古いアノテーションを削除
                        for fig in all_active_figures:
                            for ax in fig.get_axes():
                                # 既存のアノテーションをすべて削除
                                texts_to_remove = [artist for artist in ax.texts if hasattr(artist, 'is_spike_info')]
                                for artist in texts_to_remove:
                                    artist.remove()
                                # キャンバスを更新
                                fig.canvas.draw_idle()
                        
                        # クリックされた時刻から画像のインデックスを計算
                        image_index = int(x // 150)  # 150msごとに1枚の画像
                        
                        # ラベル情報の取得
                        label_info = ""
                        if image_labels is not None and image_index < len(image_labels):
                            label_info = f"\nImage Label: {image_labels[image_index]}"
                        
                        # ポップアップ表示
                        info_text = f"Time: {x:.2f}ms\nNeuron: {int(y)}{label_info}"
                        
                        # 新しいアノテーションを追加（イベントが発生したaxesにのみ）
                        annotation = event.inaxes.annotate(info_text,
                            xy=(x, y),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                        annotation.is_spike_info = True  # カスタム属性を追加
                        
                        # キャンバスを更新
                        event.inaxes.figure.canvas.draw_idle()
            
            # 既存のイベントハンドラを削除
            if hasattr(fig_spike, '_spike_click_cid'):
                fig_spike.canvas.mpl_disconnect(fig_spike._spike_click_cid)
            
            # 新しいイベントハンドラを設定
            fig_spike._spike_click_cid = fig_spike.canvas.mpl_connect('button_press_event', on_spike_click)
            
            new_figures.append(fig_spike)
        
        for monitor in selected_state_monitors:
            selected_vars = [var for var, check in var_checks.items() if check.get()]
            fig_state = state_plot(monitor, neuron_num=neuron_slider.get(), 
                                         variable_names=selected_vars, 
                                         time_end=time_window)
            new_figures.append(fig_state)
        
        # 新しいfigureがない場合は処理を終了
        if not new_figures:
            tk.messagebox.showinfo("情報", "表示するプロットがありません。モニターを選択してください。")
            return
        
        # 新しいfigureを追加し、closeイベントを設定
        for fig in new_figures:
            plot_window.figures.append(fig)
            plot_window.active_figures.add(fig)
            all_active_figures.add(fig)  # グローバルセットにも追加
            fig.canvas.mpl_connect('close_event', on_figure_closed)
        
        # 全てのアクティブなfigure間でx軸のみを同期
        def on_xlims_change(event_ax):
            if hasattr(event_ax, '_is_resetting') and event_ax._is_resetting:
                event_ax._is_resetting = False
                return
                
            current_xlim = event_ax.get_xlim()
            for fig in all_active_figures:
                for ax in fig.get_axes():
                    if ax != event_ax:
                        if ax.get_xlim() != current_xlim:
                            ax.set_xlim(current_xlim)
                            fig.canvas.draw_idle()
        
        # 新しいfigureの全axesにx軸のイベントハンドラのみを設定
        for fig in new_figures:
            for ax in fig.get_axes():
                ax.callbacks.connect('xlim_changed', on_xlims_change)
        
        # 既存のプロットと新しいプロットのx軸を揃える
        if all_active_figures:  # plot_window.active_figuresの代わりにall_active_figuresを使用
            first_fig = next(iter(all_active_figures))
            current_xlim = first_fig.get_axes()[0].get_xlim()
            for fig in all_active_figures:
                for ax in fig.get_axes():
                    ax.set_xlim(current_xlim)
                    fig.canvas.draw_idle()
        
        class CursorState:
            def __init__(self):
                self.x = None
                self.y = None
                self.source_ax = None
                self.is_panning = False
                self.update_lock = False
                self.cursor_dots = {}
                self.backgrounds = {}
        
        cursor_state = CursorState()
        
        # 各figureにcursor_stateを保存
        for fig in new_figures:
            fig.cursor_state = cursor_state
        
        # 各figureのバックグラウンドを保存
        for fig in new_figures:
            fig.canvas.draw()
            cursor_state.backgrounds[fig] = fig.canvas.copy_from_bbox(fig.bbox)
        
        def update_all_figures(x, y, source_ax):
            """全てのfigureを同期して更新"""
            if cursor_state.update_lock:
                return
            
            cursor_state.update_lock = True
            try:
                # 既存のドットとラインを削除
                for ax, artists in cursor_state.cursor_dots.items():
                    for artist in artists:
                        artist.remove()
                cursor_state.cursor_dots.clear()
                
                # 全てのウィンドウを同時に更新
                for fig in list(all_active_figures):
                    if fig in cursor_state.backgrounds:
                        fig.canvas.restore_region(cursor_state.backgrounds[fig])
                        
                        for ax in fig.get_axes():
                            if not ax.get_ylim(): # 軸が有効でない場合はスキップ
                                continue
                            
                            y_min, y_max = ax.get_ylim()
                            # source_axのスケールを使用して正規化
                            y_scaled = y * (y_max - y_min) / source_ax.get_ylim()[1]
                            
                            # 縦線を追加
                            vline = ax.axvline(x=x, color='red', alpha=0.3, linestyle='-', linewidth=1)
                            
                            # メインのドット
                            dot = ax.scatter(x, y_scaled,
                                           color='red',
                                           s=50,
                                           alpha=0.5,
                                           zorder=10)
                            
                            # 時刻表示
                            text = ax.text(x, y_max,
                                         f'{x:.2f}ms',
                                         horizontalalignment='center',
                                         verticalalignment='bottom',
                                         color='red',
                                         alpha=0.7,
                                         fontsize=8)
                            
                            cursor_state.cursor_dots[ax] = (dot, text, vline)  # vlineを追加
                            
                            # アーティストを描画
                            ax.draw_artist(vline)
                            ax.draw_artist(dot)
                            ax.draw_artist(text)
                        
                        # 即座に面を更新
                        fig.canvas.blit(fig.bbox)
                        fig.canvas.flush_events()
            finally:
                cursor_state.update_lock = False
        
        def clear_all_dots():
            """全てのドットを同期してクリア"""
            if cursor_state.update_lock:
                return
            
            cursor_state.update_lock = True
            try:
                for ax, artists in cursor_state.cursor_dots.items():
                    for artist in artists:
                        artist.remove()
                cursor_state.cursor_dots.clear()
                
                for fig in list(all_active_figures):
                    if fig in cursor_state.backgrounds:
                        fig.canvas.restore_region(cursor_state.backgrounds[fig])
                        fig.canvas.blit(fig.bbox)
                        fig.canvas.flush_events()
            finally:
                cursor_state.update_lock = False
        
        def on_mouse_move(event):
            if not event.inaxes or cursor_state.is_panning:
                if event.inaxes is None:
                    clear_all_dots()
                return
            
            # スパイク情報の表示中は処理をスキップ
            if any(hasattr(artist, 'is_spike_info') for artist in event.inaxes.texts):
                return
                
            x, y = event.xdata, event.ydata
            
            if (x != cursor_state.x or y != cursor_state.y or 
                event.inaxes != cursor_state.source_ax):
                cursor_state.x = x
                cursor_state.y = y
                cursor_state.source_ax = event.inaxes
                update_all_figures(x, y, event.inaxes)
        
        def on_mouse_leave(event):
            cursor_state.x = None
            cursor_state.y = None
            cursor_state.source_ax = None
            clear_all_dots()
        
        def on_button_press(event):
            if event.inaxes is None:  # 追加：軸の外側でクリックした場合は無視
                return
            cursor_state.is_panning = True
            clear_all_dots()
        
        def on_button_release(event):
            if event.inaxes is None:  # 追加：軸の外側でリリースした場合は無視
                return
            cursor_state.is_panning = False
            # パン操作後にバックグラウンドを更新
            for fig in list(all_active_figures):
                if fig.canvas.manager is not None:  # 追加：キンバスが有効な場合のみ更新
                    fig.canvas.draw()
                    if hasattr(fig, 'cursor_state'):
                        fig.cursor_state.backgrounds[fig] = fig.canvas.copy_from_bbox(fig.bbox)
        
        # 新しいfigureを追加し、イベントを設定
        for fig in new_figures:
            plot_window.figures.append(fig)
            plot_window.active_figures.add(fig)
            all_active_figures.add(fig)
            
            # マウスイベントのバインド前にキャンバスが有効かチェック
            if fig.canvas.manager is not None:
                fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
                fig.canvas.mpl_connect('axes_leave_event', on_mouse_leave)
                fig.canvas.mpl_connect('button_press_event', on_button_press)
                fig.canvas.mpl_connect('button_release_event', on_button_release)
                fig.canvas.mpl_connect('close_event', on_figure_closed)
                
                if fig.canvas.supports_blit:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
        
        for fig in new_figures:
            for ax in fig.get_axes():
                # プロット内の全データの最大間を取得
                max_time = 0
                for line in ax.get_lines():
                    if len(line.get_xdata()) > 0:
                        xdata = line.get_xdata()
                        # Quantity型の場合は数値に換
                        if hasattr(xdata, 'dimensions'):
                            xdata = float(xdata[-1] * 1000)  # 秒からミリ秒に変換
                        else:
                            xdata = float(max(xdata))
                        max_time = max(max_time, xdata)
                
                # Scatter plotのデータも確認
                for collection in ax.collections:
                    if len(collection.get_offsets()) > 0:
                        offsets = collection.get_offsets()
                        if hasattr(offsets, 'dimensions'):
                            offset_max = float(max(offsets[:, 0]) * 1000)  # 秒からミリ秒に変換
                        else:
                            offset_max = float(max(offsets[:, 0]))
                        max_time = max(max_time, offset_max)        
        
        # plt.show()を削除 - 余分なウィンドウの表示を防ぐ
        # 代わりに各figureを個別に表示
        for fig in new_figures:
            fig.show()
    
    # StateMonitorの変数選択用チェックボックス（StateMonitorが存在する場合のみ）
    var_checks = {}
    if state_monitors:
        var_frame = tk.LabelFrame(plot_window, text="State Variables (State Monitor)")
        var_frame.pack(fill='x', padx=10, pady=5)
        vars = ["v", "Ie", "Ii", "ge", "gi"]
        for var in vars:
            var_checks[var] = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(var_frame, text=var, variable=var_checks[var])
            cb.pack(side=tk.LEFT, padx=5)
            
        # StateMonitorのニューロン号指定用スライダ
        neuron_frame = tk.LabelFrame(plot_window, text="Neuron Number (State Monitor)")
        neuron_frame.pack(fill='x', padx=10, pady=5)
        neuron_slider = tk.Scale(neuron_frame, from_=0, to=state_monitors[0].N, orient=tk.HORIZONTAL)
        neuron_slider.set(0)
        neuron_slider.pack(fill='x', padx=5)

    
    
    # ファイル情報の表示
    info_frame = tk.LabelFrame(plot_window, text="Loaded Files")
    info_frame.pack(fill='x', padx=10, pady=5)
    
    # 統合されたモニターリストボックス
    monitor_listbox_frame = tk.Frame(info_frame)
    monitor_listbox_frame.pack(padx=5, fill='both', expand=True)
    
    monitor_listbox = tk.Listbox(monitor_listbox_frame, selectmode='multiple', height=10)
    monitor_listbox.pack(side='left', fill='both', expand=True)
    
    # スクロールバーの追加
    monitor_scrollbar = tk.Scrollbar(monitor_listbox_frame, orient="vertical")
    monitor_scrollbar.config(command=monitor_listbox.yview)
    monitor_scrollbar.pack(side='right', fill='y')
    monitor_listbox.config(yscrollcommand=monitor_scrollbar.set)
    
    # すべてのモニターをリストに追加し、デフォルト全て選択
    all_monitors = []
    for i, monitor in enumerate(spike_monitors):
        file_name = os.path.basename(file_paths[i])  # ファイル名のみを取得
        file_name = os.path.splitext(file_name)[0]   # 拡張子を除去
        monitor_listbox.insert(tk.END, f"[{file_name}] {os.path.relpath(file_paths[i])}")
        monitor_listbox.selection_set(i)
        all_monitors.append(('spike', monitor))
    
    offset = len(spike_monitors)
    for i, monitor in enumerate(state_monitors, start=offset):
        file_path_index = i - offset + len(spike_monitors)
        file_name = os.path.basename(file_paths[file_path_index])  # ファイル名のみを取得
        file_name = os.path.splitext(file_name)[0]   # 拡張子を除去
        monitor_listbox.insert(tk.END, f"[{file_name}] {os.path.relpath(file_paths[file_path_index])}")
        monitor_listbox.selection_set(i)
        all_monitors.append(('state', monitor))
    
    # プロトボタン
    plot_button = tk.Button(plot_window, text="Plot!", font=("Arial", 20, "bold"), command=plot_all, height=2, width=20)
    plot_button.pack(pady=10, expand=True, fill="both")

def open_file():
    file_paths = filedialog.askopenfilenames(
        title="Open monitor files",
        filetypes=[("Pickle files", "*.pkl")]
    )
    if file_paths:
        plot_monitor(list(file_paths))
        
def on_closing():
    plt.close('all')  # すべてのプロットウィンドウを閉じる
    root.destroy()

def close_all_plots():
    plt.close('all')  # すべてのプロットウィンドウを閉じる
    
    # すべての子ウィンドウを取得して閉じる
    for window in root.winfo_children():
        if isinstance(window, tk.Toplevel):
            window.destroy()
            


def reset_view():
    if not all_active_figures:
        return
        
    # 最初のfigureから最大の時間範囲を取得
    max_times = []
    for fig in all_active_figures:
        for ax in fig.get_axes():
            # Line plotのデータを確認
            for line in ax.get_lines():
                if len(line.get_xdata()) > 0:
                    xdata = line.get_xdata()
                    if hasattr(xdata, 'dimensions'):
                        max_time = float(xdata[-1] * 1000)  # 秒からミリ秒に変換
                    else:
                        max_time = float(max(xdata))
                    max_times.append(max_time)
            
            # Scatter plotのデータを確認
            for collection in ax.collections:
                if len(collection.get_offsets()) > 0:
                    offsets = collection.get_offsets()
                    if hasattr(offsets, 'dimensions'):
                        max_time = float(max(offsets[:, 0]) * 1000)  # 秒からミリ秒に変換
                    else:
                        max_time = float(max(offsets[:, 0]))
                    max_times.append(max_time)
    
    if max_times:
        max_time = max(max_times)
        
        # カーソル同期の一時停止
        for fig in all_active_figures:
            if hasattr(fig, 'cursor_state'):
                fig.cursor_state.update_lock = True
        
        try:
            for fig in all_active_figures:
                for ax in fig.get_axes():
                    ax._is_resetting = True
                    ax.autoscale()
                    ax.set_xlim(0, max_time)  # x軸の範囲を明示的に設定
                    fig.canvas.draw()
                
                # カーソル同期用のバックグラウンドを更新
                if hasattr(fig, 'cursor_state'):
                    fig.cursor_state.backgrounds[fig] = fig.canvas.copy_from_bbox(fig.bbox)
        finally:
            # カーソル同期の再開
            for fig in all_active_figures:
                if hasattr(fig, 'cursor_state'):
                    fig.cursor_state.update_lock = False

def change_plot_range():
    if not all_active_figures:
        tk.messagebox.showwarning("Warning", "プロットが存在しません。")
        return
    
    # 最大時間を取得
    max_times = []
    for fig in all_active_figures:
        for ax in fig.get_axes():
            # Line plotのデータを確認
            for line in ax.get_lines():
                if len(line.get_xdata()) > 0:
                    xdata = line.get_xdata()
                    if hasattr(xdata, 'dimensions'):
                        max_time = float(xdata[-1] * 1000)  # 秒からミリ秒に変換
                    else:
                        max_time = float(max(xdata))
                    max_times.append(max_time)
            
            # Scatter plotのデータを確認
            for collection in ax.collections:
                if len(collection.get_offsets()) > 0:
                    offsets = collection.get_offsets()
                    if hasattr(offsets, 'dimensions'):
                        max_time = float(max(offsets[:, 0]) * 1000)  # 秒からミリ秒に変換
                    else:
                        max_time = float(max(offsets[:, 0]))
                    max_times.append(max_time)
    
    if not max_times:
        return
    
    max_time = max(max_times)
    
    # 範囲設定ウィンドウの作成
    range_window = tk.Toplevel(root)
    range_window.title("Plot Range Settings")
    range_window.geometry("400x200")
    
    # 現在の表示範囲を取得
    current_xlim = next(iter(all_active_figures)).get_axes()[0].get_xlim()
    
    # スライダーフレーム
    slider_frame = tk.LabelFrame(range_window, text="Time Range (ms)")
    slider_frame.pack(fill='x', padx=10, pady=5)
    
    # 開始時間のスライダー
    start_frame = tk.Frame(slider_frame)
    start_frame.pack(fill='x', padx=5, pady=2)
    tk.Label(start_frame, text="Start:").pack(side=tk.LEFT)
    start_slider = tk.Scale(start_frame, from_=0, to=max_time, orient=tk.HORIZONTAL)
    start_slider.set(current_xlim[0])
    start_slider.pack(side=tk.LEFT, fill='x', expand=True)
    
    # 終了時間のスライダー
    end_frame = tk.Frame(slider_frame)
    end_frame.pack(fill='x', padx=5, pady=2)
    tk.Label(end_frame, text="End:  ").pack(side=tk.LEFT)
    end_slider = tk.Scale(end_frame, from_=0, to=max_time, orient=tk.HORIZONTAL)
    end_slider.set(current_xlim[1])
    end_slider.pack(side=tk.LEFT, fill='x', expand=True)
    
    def apply_range():
        start = start_slider.get()
        end = end_slider.get()
        if start >= end:
            tk.messagebox.showerror("Error", "開始時間は終了時間より小さくしてください。")
            return
        
        # カーソル同期の一時停止
        for fig in all_active_figures:
            if hasattr(fig, 'cursor_state'):
                fig.cursor_state.update_lock = True
        
        try:
            for fig in all_active_figures:
                for ax in fig.get_axes():
                    ax._is_resetting = True
                    ax.set_xlim(start, end)
                    fig.canvas.draw()
                
                # カーソル同期用のバックグラウンドを更新
                if hasattr(fig, 'cursor_state'):
                    fig.cursor_state.backgrounds[fig] = fig.canvas.copy_from_bbox(fig.bbox)
        finally:
            # カーソル同期の再開
            for fig in all_active_figures:
                if hasattr(fig, 'cursor_state'):
                    fig.cursor_state.update_lock = False
        
        range_window.destroy()
    
    # 適用ボタン
    apply_button = tk.Button(range_window, text="Apply", command=apply_range)
    apply_button.pack(pady=10)

root = TkinterDnD.Tk()
root.title("Monitor Visualizer")
root.geometry("800x600")

root.protocol("WM_DELETE_WINDOW", on_closing)  # 閉じるボタンが押された時のハンドラを設定

# メインのドロップゾーン（モニターファイル用）
monitor_drop_frame = tk.Frame(root, bg="lightblue", borderwidth=2, relief="groove")
monitor_drop_frame.drop_target_register(DND_FILES)
monitor_drop_frame.dnd_bind('<<Drop>>', handle_drop)
monitor_drop_frame.bind('<Button-1>', lambda e: open_file())
lb_monitor_drop = tk.Label(monitor_drop_frame, 
                          text="Drop your Monitor files here!", 
                          font=("Arial", 20, "bold"), 
                          bg="lightblue")

monitor_drop_frame.pack(fill="both", expand=True, padx=10, pady=10)
lb_monitor_drop.place(relx=0.5, rely=0.5, anchor="center")

# ラベルJSONファイル用のドロップゾーン
json_drop_frame = tk.Frame(root, bg="lightgreen", borderwidth=2, relief="groove", height=100)
json_drop_frame.drop_target_register(DND_FILES)
json_drop_frame.dnd_bind('<<Drop>>', handle_drop)
lb_json_drop = tk.Label(json_drop_frame, 
                       text="Drop label JSON file here", 
                       font=("Arial", 12, "bold"), 
                       bg="lightgreen")

json_drop_frame.pack(fill="x", padx=10, pady=5)
json_drop_frame.pack_propagate(False)  # フレームのサイズを固定
lb_json_drop.place(relx=0.5, rely=0.5, anchor="center")

# 既存のボタン
bt_change_plot_range = tk.Button(root, text="Change plot range", font=("Arial", 15, "bold"), bg="white", command=change_plot_range, height=2, width=10)
bt_change_plot_range.pack(padx=10, pady=5, fill="x")

bt_close_all = tk.Button(root, text="Close all plots", font=("Arial", 15, "bold"), bg="white", command=close_all_plots, height=2, width=10)
bt_close_all.pack(padx=10, pady=5, fill="x")

root.mainloop()
