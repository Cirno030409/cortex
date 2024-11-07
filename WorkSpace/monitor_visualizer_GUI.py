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

file_path = []

# グローバル変数としてすべてのアクティブなfigureを追跡
all_active_figures = set()

def handle_drop(event):
    global file_path
    # 複数ファイルのパスを取得
    if event.data.startswith('{'):
        # Windowsスタイルのパス処理
        files = event.data.split('} {')
        files = [f.strip('{}') for f in files]
    else:
        # Unix/Macスタイルのパス処理
        files = event.data.split(' ')
    
    # 有効なpklファイルのみを抽出
    file_paths = [f for f in files if f.endswith('.pkl')]
    
    if file_paths:
        plot_monitor(file_paths)

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
            plotter = Common_Plotter()
            fig_spike = plotter.raster_plot(selected_spike_monitors, time_end=time_window)
            new_figures.append(fig_spike)
        
        for monitor in selected_state_monitors:
            plotter = Common_Plotter()
            selected_vars = [var for var, check in var_checks.items() if check.get()]
            fig_state = plotter.state_plot(monitor, neuron_num=neuron_slider.get(), 
                                         variable_names=selected_vars, 
                                         time_end=time_window)
            new_figures.append(fig_state)
        
        # 新しいfigureを追加し、closeイベントを設定
        for fig in new_figures:
            plot_window.figures.append(fig)
            plot_window.active_figures.add(fig)
            all_active_figures.add(fig)  # グローバルセットにも追加
            fig.canvas.mpl_connect('close_event', on_figure_closed)
        
        # 全てのアクティブなfigure間でx軸を同期
        def on_xlims_change(event_ax):
            current_xlim = event_ax.get_xlim()
            # グローバルなall_active_figuresを使用して同期
            for fig in all_active_figures:
                for ax in fig.get_axes():
                    if ax != event_ax:
                        if ax.get_xlim() != current_xlim:
                            ax.set_xlim(current_xlim)
                            fig.canvas.draw_idle()
        
        # 新しいfigureの全axesにイハンドラを設定
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
                                         f'{x:.1f}ms',
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
                        
                        # 即座に画面を更新
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
            if cursor_state.is_panning or not event.inaxes:
                if event.inaxes is None:
                    clear_all_dots()
                return
            
            x, y = event.xdata, event.ydata
            
            # 座標が変化した場合のみ更新
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
            cursor_state.is_panning = True
            clear_all_dots()
        
        def on_button_release(event):
            cursor_state.is_panning = False
            # パン操作後にバックグラウンドを更新
            for fig in list(all_active_figures):
                fig.canvas.draw()
                cursor_state.backgrounds[fig] = fig.canvas.copy_from_bbox(fig.bbox)
                if hasattr(fig, 'cursor_state'):
                    fig.cursor_state.backgrounds[fig] = fig.canvas.copy_from_bbox(fig.bbox)
        
        # 新しいfigureを追加し、イベントを設定
        for fig in new_figures:
            plot_window.figures.append(fig)
            plot_window.active_figures.add(fig)
            all_active_figures.add(fig)
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
            fig.canvas.mpl_connect('axes_leave_event', on_mouse_leave)
            fig.canvas.mpl_connect('button_press_event', on_button_press)
            fig.canvas.mpl_connect('button_release_event', on_button_release)
            fig.canvas.mpl_connect('close_event', on_figure_closed)
            
            if fig.canvas.supports_blit:
                fig.canvas.draw()
                fig.canvas.flush_events()
        
        plt.show()
    
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
    
    # すべてのモニターをリストに追加し、デフォルトで全て選択
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
    
    # プロットボタン
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
            
def change_plot_range():
    # アクティブなfigureが存在しない場合は何もしない
    if not all_active_figures:
        tk.messagebox.showinfo("Info", "表示中のプロットがありません。")
        return
    
    # 現在の時間範囲を取得
    first_fig = next(iter(all_active_figures))
    current_xlim = first_fig.get_axes()[0].get_xlim()
    
    # 各プロットの最大時間を取得
    max_times = []
    for fig in all_active_figures:
        for ax in fig.get_axes():
            # Line plotのデータを確認
            lines = [line for line in ax.get_lines() if len(line.get_xdata()) > 0]
            if lines:  # データが存在する場合のみ
                max_time = float(max(max(line.get_xdata()) for line in lines))  # float型に変換
                max_times.append(max_time)
            
            # Scatter plotのデータを確認
            collections = [col for col in ax.collections if len(col.get_offsets()) > 0]
            if collections:  # データが存在する場合のみ
                for collection in collections:
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        max_time = float(max(offsets[:, 0]))  # float型��変換
                        max_times.append(max_time)
    
    # 最大表示可能時間を設定（全プロットの最小値）
    max_allowed_time = min(max_times) if max_times else current_xlim[1]
    
    # 新しいウィンドウを作成
    range_window = tk.Toplevel(root)
    range_window.title("Change Plot Range")
    range_window.geometry("400x200")
    
    # 時間範囲設定用フレーム
    time_frame = tk.LabelFrame(range_window, text=f"Time Window (max: {max_allowed_time:.1f} ms)")
    time_frame.pack(fill='x', padx=10, pady=5)
    
    # スライダーの作成
    time_slider = tk.Scale(time_frame, from_=0, to=max_allowed_time, 
                          orient=tk.HORIZONTAL, resolution=0.1)
    time_slider.set(current_xlim[1])  # 現在の表示範囲を初期値に設定
    time_slider.pack(fill='x', padx=5)
    
    def apply_range():
        new_time_window = time_slider.get()
        # すべてのプロットの時間範囲を更新
        for fig in all_active_figures:
            for ax in fig.get_axes():
                ax.set_xlim(0, new_time_window)
            fig.canvas.draw()  # draw()を呼び出して完全に再描画
            
            # カーソル同期用のバックグラウンドを更新
            if hasattr(fig, 'cursor_state'):
                fig.cursor_state.backgrounds[fig] = fig.canvas.copy_from_bbox(fig.bbox)
            
        range_window.destroy()
    
    # 適用ボタン
    apply_button = tk.Button(range_window, text="Apply", 
                            command=apply_range,
                            font=("Arial", 12, "bold"))
    apply_button.pack(pady=10)

root = TkinterDnD.Tk()
root.title("Monitor Visualizer")
root.geometry("800x600")

root.protocol("WM_DELETE_WINDOW", on_closing)  # 閉じるボタンが押された時のハンドラを設


drop_zone_frame = tk.Frame(root, bg="lightblue", borderwidth=2, relief="groove")
drop_zone_frame.drop_target_register(DND_FILES)
drop_zone_frame.dnd_bind('<<Drop>>', handle_drop)
drop_zone_frame.bind('<Button-1>', lambda e: open_file())
lb_drop_zone = tk.Label(drop_zone_frame, text="Drop your Monitor files here!", font=("Arial", 20, "bold"), bg="lightblue")

drop_zone_frame.pack(fill="both", expand=True, padx=10, pady=10)

lb_drop_zone.place(relx=0.5, rely=0.5, anchor="center")

bt_change_plot_range = tk.Button(root, text="Change plot range", font=("Arial", 15, "bold"), bg="white", command=change_plot_range, height=2, width=10)
bt_change_plot_range.pack(padx=10, pady=5, fill="x")

bt_close_all = tk.Button(root, text="Close all plots", font=("Arial", 15, "bold"), bg="white", command=close_all_plots, height=2, width=10)
bt_close_all.pack(padx=10, pady=5, fill="x")

root.mainloop()
