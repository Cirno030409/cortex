from brian2 import *
from Brian2_Framework.Tools import *
from Brian2_Framework.Plotters import *
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from Brian2_Framework.Tools import SpikeMonitorData
from Brian2_Framework.Tools import StateMonitorData

file_path = []

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
        if isinstance(monitor, (SpikeMonitorData, StateMonitorData)):
            max_simulation_time = max(max_simulation_time, float(monitor.t[-1]*1000))
    
    # プロットウィンドウの作成
    plot_window = tk.Toplevel(root)
    plot_window.title("Plot Settings")
    plot_window.geometry("550x450")
    
    
    
    # 時間設定用スライダー
    time_frame = tk.LabelFrame(plot_window, text="Time Window")
    time_frame.pack(fill='x', padx=10, pady=5)
    time_slider = tk.Scale(time_frame, from_=0, to=max_simulation_time, orient=tk.HORIZONTAL)
    time_slider.set(150)
    time_slider.pack(fill='x', padx=5)
    
    def plot_all():
        # plt.close("all")
        time_window = time_slider.get()
        figures = []  # 作成したfigureを保存するリスト
        
        if spike_monitors:
            plotter = Common_Plotter()
            fig_spike = plotter.raster_plot(spike_monitors, time_end=time_window)
            figures.append(fig_spike)
        
        if state_monitors:
            for monitor in state_monitors:
                plotter = Common_Plotter()
                selected_vars = [var for var, check in var_checks.items() if check.get()]
                fig_state = plotter.state_plot(monitor, neuron_num=neuron_slider.get(), 
                                             variable_names=selected_vars, 
                                             time_end=time_window)
                figures.append(fig_state)
        
        # 全figureのx軸を同期
        def on_xlims_change(event_ax):
            current_xlim = event_ax.get_xlim()
            for fig in figures:
                for ax in fig.get_axes():
                    if ax != event_ax:  # イベントが発生したaxes以外を更新
                        if ax.get_xlim() != current_xlim:  # すでに設定されているxlimと異なる場合のみ更新
                            ax.set_xlim(current_xlim)
                            fig.canvas.draw_idle()
        
        # 各figureの全axesにイベントハンドラを設定
        for fig in figures:
            for ax in fig.get_axes():
                ax.callbacks.connect('xlim_changed', on_xlims_change)
        
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
            
        # StateMonitorのニューロン番号指定用スライダー
        neuron_frame = tk.LabelFrame(plot_window, text="Neuron Number (State Monitor)")
        neuron_frame.pack(fill='x', padx=10, pady=5)
        neuron_slider = tk.Scale(neuron_frame, from_=0, to=state_monitors[0].N, orient=tk.HORIZONTAL)
        neuron_slider.set(0)
        neuron_slider.pack(fill='x', padx=5)

    
    
    # ファイル情報の表示
    info_frame = tk.LabelFrame(plot_window, text="Loaded Files")
    info_frame.pack(fill='x', padx=10, pady=5)
    if spike_monitors:
        tk.Label(info_frame, text=f"Spike Monitors: {len(spike_monitors)}").pack()
    if state_monitors:
        tk.Label(info_frame, text=f"State Monitors: {len(state_monitors)}").pack()
    
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

root = TkinterDnD.Tk()
root.title("Monitor Visualizer")
root.geometry("800x600")

root.protocol("WM_DELETE_WINDOW", on_closing)  # 閉じるボタンが押された時のハンドラを設定


drop_zone_frame = tk.Frame(root, bg="lightblue", borderwidth=2, relief="groove")
drop_zone_frame.drop_target_register(DND_FILES)
drop_zone_frame.dnd_bind('<<Drop>>', handle_drop)

lb_drop_zone = tk.Label(drop_zone_frame, text="Drop your Monitor files here!", font=("Arial", 20, "bold"), bg="lightblue")
bt_open_file = tk.Button(drop_zone_frame, text="Open file ↗", font=("Arial", 12), bg="white", command=open_file, height=1, width=10)

drop_zone_frame.pack(fill="both", expand=True, padx=10, pady=10)

lb_drop_zone.place(relx=0.5, rely=0.4, anchor="center")
bt_open_file.place(relx=0.5, rely=0.5, anchor="center")

bt_close_all = tk.Button(root, text="Close all plots", font=("Arial", 15, "bold"), bg="white", command=close_all_plots, height=2, width=10)
bt_close_all.pack(padx=10, pady=5, fill="x")

root.mainloop()
