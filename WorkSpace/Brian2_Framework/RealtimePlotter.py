import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.widgets import Button, Slider
from brian2 import *

class RealtimePlotter:
    def __init__(self, network, params, time_window=150):
        """
        リアルタイムプロット用のプロッターを初期化します
        
        Args:
            network (Network): モニタリング対象のネットワーク
            params (dict): ネットワークのパラメータ
            time_window (float): 表示する時間窓の幅（ミリ秒）
        """
        self.network = network
        self.params = params
        self.time_window = time_window
        self.running = False
        self.paused = False
        self.last_plot_time = 0  # 最後にプロットした時間を保存
        self.paused_time = None  # 一時停止時の時間を保存
        self.update_interval = 50  # 更新間隔(ms)
        
        # レイヤー情報の構築
        self.layer_info = [
            {
                'name': 'Input Layer',
                'n_neurons': self.params['n_inp'],
                'monitor_name': 'spikemon_inp',
                'color': 'blue'
            },
            {
                'name': 'Excitatory Layer',
                'n_neurons': self.params['n_e'],
                'monitor_name': 'spikemon_N_1',
                'color': 'red'
            },
            {
                'name': 'Inhibitory Layer',
                'n_neurons': self.params['n_i'],
                'monitor_name': 'spikemon_N_2',
                'color': 'green'
            }
        ]
        
        # プロットの初期化
        self._init_plot()
        
    def _init_plot(self):
        """
        プロットの初期化
        """
        self.fig = plt.figure(figsize=(12, 9))
        # 統計情報用のスペースを確保するためにGridSpecを調整
        plot_gs = GridSpec(3, 2, figure=self.fig, top=0.9, bottom=0.15, width_ratios=[3, 1])
        
        # メインのスパイクプロット用の軸
        self.axes = [self.fig.add_subplot(plot_gs[i, 0]) for i in range(3)]
        
        # 統計情報表示用の軸
        self.stats_ax = self.fig.add_subplot(plot_gs[:, 1])
        self.stats_ax.set_title('Network Statistics')
        self.stats_ax.axis('off')
        
        # スキャッタープロットの初期化
        self.scatter_plots = []
        for ax, layer in zip(self.axes, self.layer_info):
            scatter = ax.scatter([], [], s=2, color=layer['color'])
            self.scatter_plots.append(scatter)
            
            ax.set_ylabel("Neuron No")
            ax.set_title(f"{layer['name']} ({layer['n_neurons']} neurons)")
            ax.set_xlim(0, self.time_window)
            ax.set_ylim(-1, layer['n_neurons'])
            
        self.axes[-1].set_xlabel("Time (ms)")
        
        # コントロールパネルの設定
        ax_button = plt.axes([0.85, 0.02, 0.1, 0.04])
        self.pause_button = Button(ax_button, 'Pause')
        self.pause_button.on_clicked(self._toggle_pause)
        
        # 時間窓スライダーの設定
        ax_slider = plt.axes([0.1, 0.02, 0.6, 0.03])
        self.time_slider = Slider(
            ax_slider, 'Window (ms)', 
            50, 5000, valinit=self.time_window
        )
        self.time_slider.on_changed(self._update_time_window)
        
        # 情報表示エリア
        self.info_text = self.fig.text(0.02, 0.95, '', fontsize=8)
        
        self.fig.tight_layout()
        
    def _toggle_pause(self, event):
        """
        プロットの一時停止/再開を切り替え
        より効率的な実装
        """
        self.paused = not self.paused
        current_time = self.network.t/ms
        
        if self.paused:
            # 一時停止時の時間を保存
            self.paused_time = current_time
            self.pause_button.label.set_text('Resume')
        else:
            # 再開時に即座に更新
            self.pause_button.label.set_text('Pause')
            self.last_plot_time = current_time
            self.update_plot(force_update=True)
        
    def _update_time_window(self, val):
        """
        時間窓を更新して即座にプロットを更新
        """
        self.time_window = val
        self.update_plot(force_update=True)
        
    def update_plot(self, force_update=False):
        """
        プロットを更新
        
        Args:
            force_update (bool): 更新間隔を無視して強制的に更新するかどうか
        """
        if not self.running or self.paused:
            return
            
        try:
            current_time = self.network.t/ms
            
            # 更新間隔のチェック（強制更新の場合はスキップ）
            if not force_update and (current_time - self.last_plot_time) < self.update_interval:
                return
                
            self.last_plot_time = current_time
            time_min = max(0, current_time - self.time_window)
            time_max = current_time + 1
            
            # データ更新フラグ（実際に新しいデータがあるか確認）
            has_new_data = False
            
            for ax, layer, scatter in zip(self.axes, self.layer_info, self.scatter_plots):
                monitor = self.network[layer['monitor_name']]
                
                # 時間窓内のスパイクを抽出
                mask = (monitor.t/ms >= time_min) & (monitor.t/ms <= time_max)
                times = monitor.t[mask]/ms
                indices = monitor.i[mask]
                
                # データを更新
                if len(times) > 0:
                    has_new_data = True
                scatter.set_offsets(np.c_[times, indices])
                ax.set_xlim(time_min, time_max)
            
            # 統計情報の更新
            stats = self._calculate_statistics()
            self._update_info_text(stats)
            
            # 新しいデータがある場合のみ描画を更新
            if has_new_data or force_update:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"[WARNING] Plot update error: {str(e)}")
    
    def show(self):
        """
        プロットウィンドウを表示
        """
        self.running = True
        plt.show(block=False)
        plt.pause(0.1)
    
    def close(self):
        """
        プロッターを終了
        """
        self.running = False
        plt.close(self.fig)
        
    def _calculate_statistics(self):
        """
        ネットワーク全体の統計情報を計算
        """
        stats = {}
        current_time = self.network.t/ms
        window_start = max(0, current_time - self.time_window)
        
        for layer in self.layer_info:
            monitor = self.network[layer['monitor_name']]
            mask = (monitor.t/ms >= window_start) & (monitor.t/ms <= current_time)
            spikes = monitor.i[mask]
            spike_times = monitor.t[mask]/ms
            
            if len(spikes) > 0:
                unique_neurons = np.unique(spikes)
                spike_counts = np.bincount(spikes, minlength=layer['n_neurons'])
                isi = np.diff(spike_times[spikes == spikes[0]]) if len(spike_times) > 1 else [0]
                
                stats[layer['name']] = {
                    'total_spikes': len(spikes),
                    'active_neurons': len(unique_neurons),
                    'mean_rate': len(spikes) / (self.time_window * layer['n_neurons']) * 1000,
                    'active_ratio': len(unique_neurons) / layer['n_neurons'] * 100,
                    'max_rate': np.max(spike_counts) / (self.time_window/1000),  # 最も活発なニューロンの発火率
                    'silent_neurons': layer['n_neurons'] - len(unique_neurons),  # 発火していないニューロン数
                    'mean_isi': np.mean(isi) if len(isi) > 0 else float('inf'),  # 平均ISI
                    'cv_isi': np.std(isi)/np.mean(isi) if len(isi) > 1 else 0,  # ISIの変動係数
                    'burst_events': np.sum(isi < 10) if len(isi) > 0 else 0,  # バースト発火の回数（ISI < 10ms）
                    'most_active_neuron': np.argmax(spike_counts),  # 最も活発なニューロンのインデックス
                }
            else:
                stats[layer['name']] = {
                    'total_spikes': 0,
                    'active_neurons': 0,
                    'mean_rate': 0,
                    'active_ratio': 0,
                    'max_rate': 0,
                    'silent_neurons': layer['n_neurons'],
                    'mean_isi': float('inf'),
                    'cv_isi': 0,
                    'burst_events': 0,
                    'most_active_neuron': -1,
                }
                
        # ネットワーク全体の統計
        total_spikes = sum(s['total_spikes'] for s in stats.values())
        total_neurons = sum(layer['n_neurons'] for layer in self.layer_info)
        stats['Network Summary'] = {
            'total_spikes': total_spikes,
            'network_rate': total_spikes / (self.time_window/1000),  # ネットワーク全体の発火率
            'active_ratio': sum(s['active_neurons'] for s in stats.values()) / total_neurons * 100,
            'silent_ratio': sum(s['silent_neurons'] for s in stats.values()) / total_neurons * 100,
        }
        
        return stats

    def _update_info_text(self, stats):
        """
        統計情報テキストの更新
        """
        self.stats_ax.clear()
        self.stats_ax.axis('off')
        
        info_text = f"=== Simulation Status ===\n"
        info_text += f"Time: {self.network.t/ms:.1f} ms\n"
        info_text += f"Status: {'PAUSED' if self.paused else 'RUNNING'}\n"
        info_text += f"Window: {self.time_window:.0f} ms\n\n"
        
        # ネットワーク全体のサマリー
        net_stats = stats['Network Summary']
        info_text += f"=== Network Summary ===\n"
        info_text += f"Total Spikes: {net_stats['total_spikes']}\n"
        info_text += f"Network Rate: {net_stats['network_rate']:.1f} Hz\n"
        info_text += f"Active Neurons: {net_stats['active_ratio']:.1f}%\n"
        info_text += f"Silent Neurons: {net_stats['silent_ratio']:.1f}%\n\n"
        
        # 各層の詳細情報
        for layer_name, layer_stats in stats.items():
            if layer_name != 'Network Summary':
                info_text += f"=== {layer_name} ===\n"
                info_text += f"Spikes: {layer_stats['total_spikes']}\n"
                info_text += f"Active/Silent: {layer_stats['active_neurons']}/{layer_stats['silent_neurons']}\n"
                info_text += f"Mean Rate: {layer_stats['mean_rate']:.1f} Hz\n"
                info_text += f"Max Rate: {layer_stats['max_rate']:.1f} Hz\n"
                if layer_stats['most_active_neuron'] >= 0:
                    info_text += f"Most Active: #{layer_stats['most_active_neuron']}\n"
                if layer_stats['mean_isi'] != float('inf'):
                    info_text += f"Mean ISI: {layer_stats['mean_isi']:.1f} ms\n"
                    info_text += f"ISI CV: {layer_stats['cv_isi']:.2f}\n"
                info_text += f"Burst Events: {layer_stats['burst_events']}\n\n"
        
        self.stats_ax.text(0.05, 0.95, info_text,
                          transform=self.stats_ax.transAxes,
                          verticalalignment='top',
                          fontfamily='monospace',
                          fontsize=9)