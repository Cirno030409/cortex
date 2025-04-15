#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import threading
import time
import json
import re
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

class ProcessControlTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("プロセス管理ツール")
        
        # 初期ウィンドウサイズ（より大きく変更）
        self.root.geometry("1550x700")
        self.root.minsize(1000, 600)  # 最小サイズも大きく
        
        # デバッグ出力を無効化
        self.debug_enabled = False
        
        # プロセスを格納する辞書
        self.processes = {}
        # プロセスIDをカウント
        self.process_counter = 0
        # 設定ファイル
        self.config_file = "process_control_config.json"
        # メモを保存する辞書
        self.process_notes = {}
        # 最後に選択したディレクトリ
        self.last_directory = os.path.dirname(os.path.abspath(__file__))  # 初期ディレクトリ
        # 出力バッファの最大行数
        self.max_output_lines = 1000  # 出力バッファの最大行数
        # Treeviewの自動更新タイマー
        self.update_timer = None
        
        # ソート関連の変数
        self.sort_column = "id"  # デフォルトのソート列
        self.sort_reverse = False  # 昇順/降順のフラグ
        
        # GUIを構築
        self.create_gui()
        
        # 設定を読み込む
        self.load_config()
        
        # 現在のディレクトリのファイル一覧を表示
        self.refresh_file_list()
        
        # 終了時の処理を設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_gui(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左側フレーム（コントロールパネル）
        control_frame = ttk.LabelFrame(main_frame, text="コントロールパネル", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        # ディレクトリ選択フレーム
        dir_frame = ttk.Frame(control_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dir_frame, text="作業ディレクトリ:").pack(side=tk.LEFT)
        
        self.dir_path_var = tk.StringVar(value=os.path.abspath(os.getcwd()))
        ttk.Entry(dir_frame, textvariable=self.dir_path_var, width=30).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(dir_frame, text="参照...", command=self.browse_directory).pack(side=tk.LEFT)
        
        # ファイル一覧フレーム
        files_frame = ttk.LabelFrame(control_frame, text="ファイル一覧", padding="5")
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ファイル一覧のリストボックス
        self.files_listbox = tk.Listbox(files_frame, height=8, selectmode=tk.SINGLE)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # スクロールバー
        listbox_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=listbox_scrollbar.set)
        
        # リストボックスの選択イベント
        self.files_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        
        # 更新ボタン
        ttk.Button(files_frame, text="更新", command=self.refresh_file_list).pack(fill=tk.X, pady=5)
        
        # ファイル選択
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Pythonファイル:").pack(side=tk.LEFT)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=30).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(file_frame, text="参照...", command=self.browse_file).pack(side=tk.LEFT)
        
        # コマンドライン引数
        args_frame = ttk.Frame(control_frame)
        args_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(args_frame, text="引数:").pack(side=tk.LEFT)
        
        self.args_var = tk.StringVar()
        ttk.Entry(args_frame, textvariable=self.args_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # プロセス名
        name_frame = ttk.Frame(control_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="プロセス名:").pack(side=tk.LEFT)
        
        self.process_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.process_name_var, width=30).pack(side=tk.LEFT, padx=5)
        
        # メモ入力欄（追加）
        memo_frame = ttk.LabelFrame(control_frame, text="プロセスメモ", padding="5 5 5 0")
        memo_frame.pack(fill=tk.X, expand=False, pady=(5, 0))
        
        self.memo_var = tk.StringVar()
        ttk.Entry(memo_frame, textvariable=self.memo_var, width=40).pack(fill=tk.X, padx=5, pady=(5, 0))
        
        # 確認ダイアログを表示するかどうか
        confirm_frame = ttk.Frame(control_frame)
        confirm_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(confirm_frame, text="実行前に確認ダイアログを表示", variable=self.confirm_var).pack(anchor=tk.W)
        
        # 実行ボタンのスタイル設定
        style = ttk.Style()
        style.configure('Big.TButton', font=('Helvetica', 12, 'bold'), padding=(10, 10))
        
        # 実行ボタン
        exec_button = ttk.Button(control_frame, text="実行", command=self.start_process, style='Big.TButton')
        exec_button.pack(fill=tk.X, pady=15, ipady=8)  # ipadyで内部パディングを増やして高さを拡大
        
        # 右側フレーム（プロセスリストとメモ）
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # プロセスリスト
        list_frame = ttk.LabelFrame(right_frame, text="実行中プロセス", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # 進捗バーインラインフレーム
        self.progress_frames = {}
        
        # Treeviewの列を定義（プロセス名カラムを削除、ファイルカラムを先頭に）
        columns = ("id", "file", "args", "status", "progress_bar", "progress_text", "start_time", "pid", "memo")
        self.process_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10, selectmode="extended")
        
        # 各列の見出しを設定
        self.process_tree.heading("id", text="ID", command=lambda: self.sort_treeview("id", int))
        self.process_tree.heading("file", text="ファイル", command=lambda: self.sort_treeview("file", str))
        self.process_tree.heading("args", text="引数", command=lambda: self.sort_treeview("args", str))
        self.process_tree.heading("status", text="状態", command=lambda: self.sort_treeview("status", str))
        self.process_tree.heading("progress_bar", text="進捗バー")  # 進捗バーはソート対象外
        self.process_tree.heading("progress_text", text="進捗詳細", command=lambda: self.sort_treeview("progress_text", str))
        self.process_tree.heading("start_time", text="開始時間", command=lambda: self.sort_treeview("start_time", str))
        self.process_tree.heading("pid", text="PID", command=lambda: self.sort_treeview("pid", int))
        self.process_tree.heading("memo", text="メモ", command=lambda: self.sort_treeview("memo", str))
        
        # 列の幅を設定
        self.process_tree.column("id", width=20)
        self.process_tree.column("file", width=180)  # ファイル列の幅を少し広げる
        self.process_tree.column("args", width=120)
        self.process_tree.column("status", width=70)
        self.process_tree.column("progress_bar", width=250)
        self.process_tree.column("progress_text", width=140)
        self.process_tree.column("start_time", width=140)
        self.process_tree.column("pid", width=60)
        self.process_tree.column("memo", width=150)
        
        # スクロールバーを追加
        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.process_tree.yview)
        self.process_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Treeviewとスクロールバーを配置
        self.process_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 選択したアイテムの操作
        self.process_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        
        # ボタンフレーム
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # 操作ボタン
        ttk.Button(button_frame, text="停止", command=self.stop_process).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="再起動", command=self.restart_process).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="出力を表示", command=self.show_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="メモを編集", command=self.edit_memo).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="選択プロセスを削除", command=self.remove_selected_processes).pack(side=tk.LEFT, padx=5)
        
        # 選択中プロセスの詳細プログレスバー表示フレーム
        self.progress_frame = ttk.LabelFrame(right_frame, text="選択中プロセスの詳細進捗", padding="10")
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        # 進捗情報表示ラベル
        self.progress_info_var = tk.StringVar(value="プロセスを選択してください")
        ttk.Label(self.progress_frame, textvariable=self.progress_info_var).pack(anchor=tk.W, pady=5)
        
        # 進捗バー
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 進捗の詳細テキスト
        self.progress_detail_var = tk.StringVar(value="")
        ttk.Label(self.progress_frame, textvariable=self.progress_detail_var).pack(anchor=tk.W, pady=5)
        
        # Treeviewの更新タイマーを開始
        self.start_progress_update_timer()
    
    def browse_file(self):
        filetypes = (
            ('Pythonファイル', '*.py'),
            ('すべてのファイル', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='Pythonファイルを選択',
            initialdir=self.last_directory,
            filetypes=filetypes
        )
        if filename:
            self.file_path_var.set(filename)
            # 選択したディレクトリを記憶
            self.last_directory = os.path.dirname(filename)
            # デフォルトのプロセス名としてファイル名（拡張子なし）を設定
            process_name = os.path.basename(filename)
            process_name = os.path.splitext(process_name)[0]
            self.process_name_var.set(process_name)
    
    def browse_directory(self):
        """ディレクトリ選択ダイアログを表示"""
        directory = filedialog.askdirectory(
            title='作業ディレクトリを選択',
            initialdir=self.dir_path_var.get()
        )
        if directory:
            self.dir_path_var.set(directory)
            self.last_directory = directory
            # ディレクトリ内のファイル一覧を更新
            self.refresh_file_list()
    
    def refresh_file_list(self):
        """現在のディレクトリ内のPythonファイル一覧を更新"""
        # リストボックスをクリア
        self.files_listbox.delete(0, tk.END)
        
        # 現在のディレクトリパスを取得
        directory = self.dir_path_var.get()
        
        if not os.path.isdir(directory):
            messagebox.showerror("エラー", f"ディレクトリが存在しません: {directory}")
            return
            
        # ディレクトリ内のPythonファイルを取得
        try:
            files = [f for f in os.listdir(directory) if f.endswith('.py')]
            files.sort()  # アルファベット順にソート
            
            # リストボックスに追加
            for file in files:
                self.files_listbox.insert(tk.END, file)
                
            # ファイルが見つからない場合
            if not files:
                self.files_listbox.insert(tk.END, "Pythonファイルが見つかりません")
                
        except Exception as e:
            messagebox.showerror("エラー", f"ディレクトリの読み取りに失敗しました: {e}")
    
    def on_file_select(self, event):
        """ファイル一覧からファイルが選択されたときの処理"""
        # 選択されたインデックスを取得
        selected_indices = self.files_listbox.curselection()
        if not selected_indices:
            return
            
        # 選択されたファイル名を取得
        selected_file = self.files_listbox.get(selected_indices[0])
        
        # "Pythonファイルが見つかりません" メッセージの場合は何もしない
        if selected_file == "Pythonファイルが見つかりません":
            return
            
        # 完全なパスを構築
        directory = self.dir_path_var.get()
        full_path = os.path.join(directory, selected_file)
        
        # ファイルパスを設定
        self.file_path_var.set(full_path)
        
        # デフォルトのプロセス名としてファイル名（拡張子なし）を設定
        process_name = os.path.splitext(selected_file)[0]
        self.process_name_var.set(process_name)
    
    def start_process(self):
        # ファイルパスを取得
        file_path = self.file_path_var.get()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("エラー", "有効なPythonファイルを選択してください。")
            return
            
        # コマンドライン引数
        args = self.args_var.get().strip().split()
        
        # プロセス名（入力があれば使用、なければファイル名を使用）
        process_name = self.process_name_var.get()
        if not process_name:
            process_name = os.path.basename(file_path)
            
        # メモを取得
        memo = self.memo_var.get()
        
        # 確認ダイアログが有効ならば、実行前に確認
        if self.confirm_var.get():
            if not messagebox.askyesno("確認", f"以下のプログラムを実行しますか？\n\nファイル: {file_path}\n引数: {' '.join(args)}\nプロセス名: {process_name}"):
                return
        
        # プロセスを起動
        try:
            process = subprocess.Popen(
                [sys.executable, file_path] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
        except Exception as e:
            messagebox.showerror("エラー", f"プロセスの起動に失敗しました: {e}")
            return
            
        # プロセスIDを割り当て
        self.process_counter += 1
        process_id = self.process_counter
        
        # 実行開始時間
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # プロセス情報を保存
        self.processes[process_id] = {
            "process": process,
            "file": file_path,
            "args": args,
            "name": process_name,
            "start_time": start_time,
            "output": [],
            "status": "実行中",
            "progress": "0%",
            "progress_percent": 0,
            "memo": memo,
            "manually_stopped": False  # 手動停止フラグを追加
        }
        
        # メモを記録
        self.process_notes[process_id] = memo
        
        # Treeviewのアイテムとして追加
        self.process_tree.insert(
            "",
            "end",
            iid=str(process_id),
            values=(
                process_id,
                os.path.basename(file_path),  # ファイル名を表示
                " ".join(args) if args else "",
                "実行中",
                "",  # 進捗バー列
                "",  # 進捗テキスト列
                start_time,
                process.pid,
                memo
            )
        )
        
        # 現在のソート順を適用（方向は変えない）
        if hasattr(self, 'sort_column') and self.sort_column:
            self.sort_treeview(self.sort_column, int if self.sort_column in ["id", "pid"] else str, toggle_direction=False)
        
        # 出力読み取りスレッドを開始
        t = threading.Thread(target=self.read_output, args=(process_id,))
        t.daemon = True
        t.start()
        
        # プロセス監視スレッドを開始
        t2 = threading.Thread(target=self.monitor_process, args=(process_id,))
        t2.daemon = True
        t2.start()
    
    def read_output(self, process_id):
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        process = process_info["process"]
        
        # プロセスの出力を読み取る
        for line in iter(process.stdout.readline, ''):
            if line:
                try:
                    # シンプルにテキストを追加
                    self.processes[process_id]["output"].append(line)
                    
                    # tqdmプログレスバーを検出して保存（検出条件を拡大）
                    # tqdmのパターン検出を強化
                    progress_detected = False
                    
                    # 各行を処理
                    stripped_line = line.strip()
                    
                    # パターン1: パーセント表示
                    if '%' in line:
                        progress_detected = True
                    
                    # パターン2: it/s (iterations per second)
                    elif 'it/s' in line:
                        progress_detected = True
                    
                    # パターン3: [経過時間<残り時間] 形式
                    elif re.search(r'\[\d+:\d+<\d+:\d+', line):
                        progress_detected = True
                    
                    # パターン4: 一般的な進行状況表示 (例: 10/100)
                    elif re.search(r'\d+/\d+', line):
                        progress_detected = True
                    
                    # tqdmのプログレスバー形式（例：34%|███▍      | 76/50 [00:00<00:00, 166.15it/s]）
                    if progress_detected:
                        # tqdmプログレスと思われる行
                        progress_text = line.strip()
                        if progress_text:
                            # 前の行の内容と比較して、変更があれば更新
                            old_progress = self.processes[process_id].get("progress", "")
                            if progress_text != old_progress:
                                self.processes[process_id]["progress"] = progress_text
                                # 数値の進捗度（パーセント）を抽出
                                percent = self.extract_progress_percent(progress_text)
                                if percent is not None:
                                    self.processes[process_id]["progress_percent"] = percent
                    
                    # バッファが大きくなりすぎないように制限
                    if len(self.processes[process_id]["output"]) > self.max_output_lines:
                        # 前半を削除
                        self.processes[process_id]["output"] = self.processes[process_id]["output"][-self.max_output_lines:]
                        
                except UnicodeDecodeError:
                    # 文字コードエラーが発生した場合、代替文字に置き換える
                    safe_line = line.encode('utf-8', errors='replace').decode('utf-8')
                    self.processes[process_id]["output"].append(f"[文字コードエラー] {safe_line}")
    
    def monitor_process(self, process_id):
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        process = process_info["process"]
        
        # プロセスの原プロセスIDを保存（プロセスが置き換えられた場合に備えて）
        original_pid = process.pid
        
        # プロセスが終了するまで待機
        process.wait()
        
        # プロセスが既に再起動などで置き換えられていないか確認
        if process_id in self.processes and self.processes[process_id]["process"].pid == original_pid:
            # GUIの更新はメインスレッドで行う
            self.root.after(100, self.update_process_status, process_id)
        # 別のプロセスに置き換えられている場合は何もしない（新しいプロセスの監視スレッドに任せる）
    
    def update_process_status(self, process_id):
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        process = process_info["process"]
        
        # 終了コードを取得
        return_code = process.poll()
        
        # 手動停止フラグをチェック - これを最初に行う
        if process_info.get("manually_stopped", False):
            # 手動停止または再起動準備中の場合
            self.processes[process_id]["status"] = "手動停止"
            self.processes[process_id]["progress"] = "停止済み"
            
            # 通知しない - 手動停止の場合は通知不要
            notify = False
        elif return_code == 0:
            # 正常終了の場合
            self.processes[process_id]["status"] = "正常終了"
            self.processes[process_id]["progress"] = "完了"
            notify = False
        else:
            # 異常終了の場合
            self.processes[process_id]["status"] = f"異常終了 ({return_code})"
            self.processes[process_id]["progress"] = "エラー"
            notify = True  # 通知が必要かどうかのフラグ
        
        # Treeviewを更新
        self.process_tree.item(
            str(process_id),
            values=(
                process_id,
                os.path.basename(process_info["file"]),
                " ".join(process_info["args"]) if process_info["args"] else "",
                self.processes[process_id]["status"],
                self.processes[process_id]["progress"],
                "",  # 進捗テキスト
                process_info["start_time"],
                process.pid,
                process_info["memo"]
            )
        )
        
        # 異常終了の場合のみ通知（手動停止でない場合）
        if notify:
            self.notify_abnormal_termination(process_id, return_code)
    
    def on_tree_select(self, event):
        """ツリー項目選択時の処理"""
        # 選択されたアイテムのIDを取得
        selected_items = self.process_tree.selection()
        if not selected_items:
            # 選択がない場合はプログレスバー表示をリセット
            self.progress_info_var.set("プロセスを選択してください")
            self.progress_detail_var.set("")
            self.progress_bar["value"] = 0
            return
            
        process_id = int(selected_items[0])
        
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        
        # プロセス名と状態を表示（ファイル名をプロセス名の代わりに使用）
        self.progress_info_var.set(f"{os.path.basename(process_info['file'])} ({process_info['status']})")
        
        # 進捗の詳細を表示
        progress = process_info.get("progress", "")
        self.progress_detail_var.set(progress)
        
        # プログレスバーの更新
        percent = process_info.get("progress_percent", 0)
        self.progress_bar["value"] = percent
    
    def stop_process(self):
        # 選択されたアイテムのIDを取得
        selected_items = self.process_tree.selection()
        if not selected_items:
            messagebox.showinfo("情報", "停止するプロセスを選択してください。")
            return
        
        # 複数選択されている場合の確認
        if len(selected_items) > 1:
            if not messagebox.askyesno("確認", f"選択された {len(selected_items)} 個のプロセスをすべて停止しますか？"):
                return
        
        # 停止処理に成功したプロセス数をカウント
        stopped_count = 0
        
        # 選択されたすべてのプロセスに対して処理
        for item_id in selected_items:
            process_id = int(item_id)
            
            if process_id not in self.processes:
                continue
                
            process_info = self.processes[process_id]
            process = process_info["process"]
            
            # プロセスが実行中かどうかをチェック
            if process.poll() is None:
                # 単一選択の場合のみ個別確認
                if len(selected_items) == 1:
                    if not messagebox.askyesno("確認", f"実行中のプロセス '{process_info['name']}' を停止しますか？"):
                        return
                
                # 手動停止フラグを設定
                self.processes[process_id]["manually_stopped"] = True
                    
                # プロセスを終了
                process.terminate()
                
                # プロセスが終了するのを待つ
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 強制終了
                    process.kill()
                
                # 状態を更新
                self.processes[process_id]["status"] = "停止"
                self.processes[process_id]["progress"] = "停止"  # 進捗も停止
                
                # Treeviewを更新
                self.process_tree.item(
                    str(process_id),
                    values=(
                        process_id,
                        os.path.basename(process_info["file"]),
                        " ".join(process_info["args"]) if process_info["args"] else "",
                        "停止",
                        "░░░░░░░░░░░░░░░░░ 0%",  # 進捗バーをリセット
                        "停止",  # 進捗表示も停止
                        process_info["start_time"],
                        process.pid,
                        process_info["memo"]
                    )
                )
                
                stopped_count += 1
            else:
                # 単一選択の場合のみメッセージを表示
                if len(selected_items) == 1:
                    messagebox.showinfo("情報", f"プロセス '{process_info['name']}' はすでに終了しています。")
        
    def restart_process(self):
        # 選択されたアイテムのIDを取得
        selected_items = self.process_tree.selection()
        if not selected_items:
            messagebox.showinfo("情報", "再起動するプロセスを選択してください。")
            return
        
        # 複数選択されている場合の確認
        if len(selected_items) > 1:
            if not messagebox.askyesno("確認", f"選択された {len(selected_items)} 個のプロセスをすべて再起動しますか？"):
                return
        
        # 再起動処理に成功したプロセス数をカウント
        restarted_count = 0
        
        # 選択されたすべてのプロセスに対して処理
        for item_id in selected_items:
            process_id = int(item_id)
            
            if process_id not in self.processes:
                continue
                
            process_info = self.processes[process_id]
            
            # 手動停止フラグを設定（異常終了通知を防ぐため）
            # プロセスを停止する前に先にフラグを設定する
            self.processes[process_id]["manually_stopped"] = True
            
            # 現在のプロセスを停止（もし実行中なら）
            process = process_info["process"]
            if process.poll() is None:
                # 単一選択の場合のみ個別確認
                if len(selected_items) == 1:
                    if not messagebox.askyesno("確認", f"実行中のプロセス '{process_info['name']}' を再起動しますか？"):
                        # キャンセルされた場合、手動停止フラグをリセット
                        self.processes[process_id]["manually_stopped"] = False
                        return
                
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            # 新しいプロセスを起動
            cmd = ["python", process_info["file"]] + process_info["args"]
            try:
                # 環境変数を設定（UTF-8を使用）
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                
                new_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',  # 明示的にUTF-8を指定
                    errors='replace',  # エラー発生時は代替文字に置き換える
                    bufsize=1,
                    universal_newlines=True,
                    env=env,  # 環境変数を渡す
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                # 現在の時間
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # プロセス情報を更新（旧プロセス情報を保存してから上書き）
                old_process = self.processes[process_id].get("process", None)
                
                # 一時的にプロセス情報を辞書から削除して、古いプロセスのmonitor_processが
                # 終了時に誤った更新をしないようにする
                temp_process_info = self.processes[process_id].copy()
                del self.processes[process_id]
                
                # 少し待機して、古いプロセスの監視スレッドが終了するのを待つ
                time.sleep(0.1)
                
                # 新しいプロセス情報を設定
                self.processes[process_id] = temp_process_info
                self.processes[process_id]["process"] = new_process
                self.processes[process_id]["status"] = "実行中"
                self.processes[process_id]["progress"] = ""  # 進捗をリセット
                self.processes[process_id]["progress_percent"] = 0  # 進捗パーセントもリセット
                self.processes[process_id]["start_time"] = now
                self.processes[process_id]["output"] = []
                self.processes[process_id]["pid"] = new_process.pid
                self.processes[process_id]["manually_stopped"] = False  # 手動停止フラグをリセット
                
                # Treeviewを更新（タグもリセット）
                self.process_tree.item(
                    str(process_id),
                    values=(
                        process_id,
                        os.path.basename(process_info["file"]),
                        " ".join(process_info["args"]) if process_info["args"] else "",
                        "実行中",
                        "",  # 進捗バー表示をリセット
                        "",  # 進捗テキスト表示をリセット
                        now,
                        new_process.pid,
                        process_info["memo"]
                    ),
                    tags=()  # エラーを示すタグをクリア
                )
                
                # 出力読み取りスレッドを開始
                t = threading.Thread(target=self.read_output, args=(process_id,))
                t.daemon = True
                t.start()
                
                # プロセス監視スレッドを開始
                t2 = threading.Thread(target=self.monitor_process, args=(process_id,))
                t2.daemon = True
                t2.start()
                
                restarted_count += 1
                
            except Exception as e:
                # 単一選択の場合のみエラーメッセージを表示
                if len(selected_items) == 1:
                    messagebox.showerror("エラー", f"プロセスの再起動に失敗しました: {e}")
        
        # 複数のプロセスを再起動した場合、結果を表示
        if len(selected_items) > 1:
            messagebox.showinfo("再起動完了", f"{restarted_count} 個のプロセスを再起動しました。")
    
    def show_output(self):
        # 選択されたアイテムのIDを取得
        selected_items = self.process_tree.selection()
        if not selected_items:
            messagebox.showinfo("情報", "出力を表示するプロセスを選択してください。")
            return
        
        # 複数選択されている場合は最初のプロセスのみ表示
        if len(selected_items) > 1:
            messagebox.showinfo("情報", f"{len(selected_items)} 個のプロセスが選択されていますが、出力表示は最初の1つのみ表示します。")
            
        process_id = int(selected_items[0])
        
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        
        # 出力表示ウィンドウを作成
        output_window = tk.Toplevel(self.root)
        output_window.title(f"プロセス出力: {process_info['name']}")
        output_window.geometry("800x600")
        output_window.transient(self.root)  # メインウィンドウに対する子ウィンドウに設定
        
        # メインフレーム
        main_frame = ttk.Frame(output_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 情報フレーム
        info_frame = ttk.LabelFrame(main_frame, text="プロセス情報", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # プロセス情報を表示
        ttk.Label(info_frame, text=f"ファイル: {process_info['file']}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"引数: {' '.join(process_info['args'] if process_info['args'] else ['(なし)'])}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"開始時間: {process_info['start_time']}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"ステータス: {process_info['status']}").pack(anchor=tk.W)
        
        # 出力テキストエリア - LabelFrameを使用して視覚的に区別
        output_frame = ttk.LabelFrame(main_frame, text="出力", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # スクロールバーを明示的に作成
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 出力テキストボックスと明示的にスクロールバーを関連付け
        output_text = tk.Text(output_frame, wrap=tk.WORD, font=("Consolas", 10), 
                              yscrollcommand=scrollbar.set)
        output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=output_text.yview)
        
        # コントロールフレーム
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # ユーザーによる手動スクロールを検出するための変数
        user_scrolled = False
        last_known_position = 0.0
        scroll_threshold = 0.01  # スクロール検出のしきい値
        
        # 自動スクロールのチェックボックス
        auto_scroll_var = tk.BooleanVar(value=True)
        
        # 自動スクロールチェックボックスの変更イベント
        def on_auto_scroll_changed():
            nonlocal user_scrolled
            if auto_scroll_var.get():
                # 自動スクロールが有効になったら、次の更新時に末尾に移動するようリセット
                user_scrolled = False
                output_text.see(tk.END)  # 即座に末尾へ移動
        
        # チェックボックスを作成（1つだけ）
        auto_scroll_checkbox = ttk.Checkbutton(control_frame, text="自動スクロール", 
                                             variable=auto_scroll_var, 
                                             command=on_auto_scroll_changed)
        auto_scroll_checkbox.pack(side=tk.LEFT)
        
        # 閉じるボタン
        ttk.Button(control_frame, text="閉じる", command=output_window.destroy).pack(side=tk.RIGHT, padx=5)
        
        # ユーザーがスクロールしたときのイベントハンドラ
        def on_scroll_change(*args):
            nonlocal user_scrolled, last_known_position
            current_view = output_text.yview()
            
            # テキストの最後が見えているかどうかを確認
            at_end = (current_view[1] >= 0.99)
            
            # スクロール位置が前回から変化したか、かつ、最後まで表示されていない場合
            if abs(current_view[0] - last_known_position) > scroll_threshold and not at_end:
                user_scrolled = True
            
            # 最後まで表示されている場合はユーザースクロールフラグをリセット
            if at_end:
                user_scrolled = False
            
            last_known_position = current_view[0]
        
        # スクロールイベントにハンドラを登録（多様なスクロールイベントをカバー）
        output_text.bind("<MouseWheel>", lambda event: on_scroll_change())
        output_text.bind("<Button-4>", lambda event: on_scroll_change())  # Linuxでのホイールアップ
        output_text.bind("<Button-5>", lambda event: on_scroll_change())  # Linuxでのホイールダウン
        output_text.bind("<Prior>", lambda event: on_scroll_change())     # PageUp
        output_text.bind("<Next>", lambda event: on_scroll_change())      # PageDown
        scrollbar.bind("<B1-Motion>", lambda event: on_scroll_change())
        scrollbar.bind("<ButtonRelease-1>", lambda event: on_scroll_change())
        
        # 出力を更新する関数
        def update_output():
            nonlocal user_scrolled
            
            # 現在のスクロール位置を記憶
            current_position = output_text.yview()
            was_at_end = (current_position[1] >= 0.99)
            
            # テキストの内容と長さを取得
            old_content = output_text.get(1.0, tk.END)
            old_length = len(old_content)
            
            # テキストが変更されないようにロック
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            
            # エラー処理を追加して文字化けを防止
            try:
                if process_id in self.processes and self.processes[process_id]["output"]:
                    new_content = "".join(self.processes[process_id]["output"])
                    output_text.insert(tk.END, new_content)
                    
                    # 内容が変わっていない場合はスクロール状態を保持
                    if len(new_content) == old_length and new_content == old_content:
                        content_changed = False
                    else:
                        content_changed = True
                else:
                    output_text.insert(tk.END, "出力はありません。")
                    content_changed = True
            except UnicodeEncodeError as e:
                # 問題のある文字をエスケープして表示
                safe_output = []
                for line in self.processes[process_id]["output"]:
                    try:
                        safe_output.append(line)
                    except UnicodeEncodeError:
                        safe_output.append(line.encode('utf-8', errors='replace').decode('utf-8'))
                        
                output_text.insert(tk.END, "".join(safe_output))
                content_changed = True
            
            # スクロール位置の決定
            # 1. 自動スクロールがオンで、ユーザーが手動スクロールしていない、または元々末尾にいた場合は末尾へ
            # 2. それ以外の場合は現在のスクロール位置を維持
            should_auto_scroll = (auto_scroll_var.get() and (not user_scrolled or was_at_end))
            
            if should_auto_scroll and content_changed:
                output_text.see(tk.END)  # 自動スクロール
            else:
                # 以前のスクロール位置を復元（可能な限り）
                try:
                    output_text.yview_moveto(current_position[0])
                except:
                    pass
            
            # プロセスがまだ実行中なら更新を続ける
            if process_id in self.processes and self.processes[process_id]["process"].poll() is None:
                output_window.after(100, update_output)  # 更新頻度を調整
        
        # 初回実行
        update_output()
        
        # ウィンドウのフォーカスを設定
        output_window.focus_set()
        output_text.config(state=tk.NORMAL)  # 編集可能にする
    
    def edit_memo(self):
        # 選択されたアイテムのIDを取得
        selected_items = self.process_tree.selection()
        if not selected_items:
            messagebox.showinfo("情報", "メモを編集するプロセスを選択してください。")
            return
        
        # 複数選択されている場合は最初のプロセスのみ編集
        if len(selected_items) > 1:
            messagebox.showinfo("情報", f"{len(selected_items)} 個のプロセスが選択されていますが、メモ編集は最初の1つのみ行います。")
            
        process_id = int(selected_items[0])
        
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        
        # メモ編集ダイアログを作成
        memo_dialog = tk.Toplevel(self.root)
        memo_dialog.title(f"メモ編集: {process_info['name']}")
        memo_dialog.geometry("400x200")
        memo_dialog.transient(self.root)  # メインウィンドウに対する子ウィンドウに設定
        memo_dialog.grab_set()  # モーダルダイアログに設定
        
        # メモ入力欄
        ttk.Label(memo_dialog, text="メモ:").pack(anchor=tk.W, padx=10, pady=5)
        
        memo_text = tk.Text(memo_dialog, height=5, width=40)
        memo_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        memo_text.insert(tk.END, process_info["memo"])
        
        # ボタンフレーム
        button_frame = ttk.Frame(memo_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 保存ボタン
        def save_memo():
            new_memo = memo_text.get(1.0, tk.END).strip()
            process_info["memo"] = new_memo
            
            # Treeviewを更新
            self.process_tree.item(
                str(process_id),
                values=(
                    process_id,
                    os.path.basename(process_info["file"]),
                    " ".join(process_info["args"]) if process_info["args"] else "",
                    process_info["status"],
                    process_info["progress"],
                    process_info.get("progress_text", ""),  # 進捗テキスト
                    process_info["start_time"],
                    process_info["process"].pid,
                    new_memo
                )
            )
            
            # 設定を保存
            self.save_config()
            
            memo_dialog.destroy()
            messagebox.showinfo("成功", "メモが保存されました。")
        
        ttk.Button(button_frame, text="保存", command=save_memo).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="キャンセル", command=memo_dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def save_config(self):
        # 現在の設定を保存
        config = {
            "process_notes": {k: self.processes[k]["memo"] for k in self.processes if "memo" in self.processes[k]},
            "recent_files": list(set([p["file"] for p in self.processes.values() if isinstance(p["file"], str) and p["file"]])),
            "last_directory": self.last_directory,
            "work_directory": self.dir_path_var.get()
        }
        
        try:
            with open("process_control_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.debug_log(f"設定の保存に失敗しました: {e}")
    
    def load_config(self):
        # 設定を読み込む
        try:
            if os.path.exists("process_control_config.json"):
                with open("process_control_config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    
                # プロセスノートを読み込む
                self.process_notes = config.get("process_notes", {})
                
                # 最後に選択したディレクトリを設定
                self.last_directory = config.get("last_directory", os.path.dirname(os.path.abspath(__file__)))
                
                # 作業ディレクトリを設定
                work_dir = config.get("work_directory", os.path.dirname(os.path.abspath(__file__)))
                if os.path.isdir(work_dir):
                    self.dir_path_var.set(work_dir)
        except Exception as e:
            self.debug_log(f"設定の読み込みに失敗しました: {e}")
    
    def on_closing(self):
        # アプリケーション終了時の処理
        
        # 実行中のプロセスがあるか確認
        running_processes = False
        for process_id, process_info in self.processes.items():
            process = process_info["process"]
            if process.poll() is None:  # Noneの場合はプロセスが実行中
                running_processes = True
                break
        
        # 実行中のプロセスがある場合のみ確認ダイアログを表示
        should_close = True  # デフォルトで終了する
        if running_processes:
            should_close = messagebox.askokcancel("終了確認", "終了すると、すべての実行中プロセスが停止されます。\n終了しますか？")
        
        if should_close:
            # 更新タイマーを停止
            if self.update_timer:
                self.root.after_cancel(self.update_timer)
                
            # 実行中のプロセスをすべて停止
            for process_id, process_info in list(self.processes.items()):
                process = process_info["process"]
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
            
            # 設定を保存
            self.save_config()
            
            # アプリケーションを終了
            self.root.destroy()
    
    def start_progress_update_timer(self):
        """Treeviewの進捗情報を定期的に更新するタイマーを開始"""
        self.update_progress_display()
        # 更新頻度を上げる（200msに変更）
        self.update_timer = self.root.after(200, self.start_progress_update_timer)
    
    def update_progress_display(self):
        """全プロセスの進捗表示を更新"""
        # 選択されているプロセスを取得
        selected_items = self.process_tree.selection()
        selected_process_id = int(selected_items[0]) if selected_items else None
        
        # すべてのプロセスの進捗表示を更新
        for process_id, process_info in self.processes.items():
            if process_info["process"].poll() is None:  # プロセスが実行中
                # 進捗情報があれば更新
                progress = process_info.get("progress", "")
                percent = process_info.get("progress_percent", 0)
                
                # tqdmの出力から必要な部分だけを抽出
                if progress:
                    # 改行とキャリッジリターンを取り除く
                    progress = progress.replace('\r', '').replace('\n', '').strip()
                    
                    # 表示しやすいように整形
                    if '%' in progress:
                        # パーセント表示を探す (例: 76%|████████▏   | 76/100 [00:07<00:02,  9.22it/s])
                        parts = progress.split('|')
                        if len(parts) >= 2:
                            percent_part = parts[0].strip()
                            speed_part = parts[-1].strip() if len(parts) > 2 else ""
                            
                            # 簡潔な表示にする
                            progress = f"{percent_part}"
                            if speed_part and 'it/s' in speed_part:
                                progress += f" {speed_part}"
                    
                    # または [時間/残り時間] 形式の処理
                    elif '[' in progress and ']' in progress:
                        # [00:01<00:59, 1.67it/s] のようなフォーマット
                        progress = progress.strip()
                    
                    # Progress: N/M 形式の処理
                    elif 'Progress:' in progress and '/' in progress:
                        match = re.search(r'Progress: (\d+)/(\d+)', progress)
                        if match:
                            current, total = match.groups()
                            percent = int(int(current) / int(total) * 100)
                            progress = f"{percent}% ({current}/{total})"
                
                # プログレスバー表示の更新（選択中のプロセスのみ）
                if selected_process_id == process_id:
                    # プロセス名と状態を表示
                    self.progress_info_var.set(f"{os.path.basename(process_info['file'])} ({process_info['status']})")
                    
                    # 進捗の詳細を表示
                    self.progress_detail_var.set(progress)
                    
                    # プログレスバーの更新
                    self.progress_bar["value"] = percent
                
                # Treeviewを更新（進捗情報のみ）
                try:
                    current_values = self.process_tree.item(str(process_id), "values")
                    if current_values:
                        # 現在の値を取得して進捗情報だけ更新
                        new_values = list(current_values)
                        
                        # プログレスバー列にUnicodeの棒文字を使って視覚的な進捗バーを表示
                        if percent is not None:
                            # Unicode Block Elementsを使って進捗バーを表現
                            bar_length = 17  # バーの長さを短縮（20→17）
                            filled_length = int(bar_length * percent / 100)
                            progress_bar = '█' * filled_length + '░' * (bar_length - filled_length)
                            progress_text = f"{progress}"
                            
                            # Treeviewの表示を更新
                            new_values[4] = f"{progress_bar} {percent:3d}%"  # 右寄せパーセント表示
                            new_values[5] = progress_text                # 6番目（インデックス5）が進捗テキスト列
                        else:
                            # 進捗情報がない場合は空のバーを表示
                            new_values[4] = "░░░░░░░░░░░░░░░░░ 0%"
                            new_values[5] = "進行状況なし"
                        
                        self.process_tree.item(
                            str(process_id),
                            values=tuple(new_values)
                        )
                except Exception as e:
                    # エラーは無視
                    pass
        
        # 選択プロセスが変更された場合の処理
        if selected_process_id is not None:
            self.on_tree_select(None)
        
        # 更新が完了したら現在のソート順を維持（方向は変えない）
        if hasattr(self, 'sort_column') and self.sort_column and self.process_tree.get_children():
            self.sort_treeview(self.sort_column, int if self.sort_column in ["id", "pid"] else str, toggle_direction=False)
    
    def extract_progress_percent(self, progress_text):
        """進捗テキストからパーセント値を抽出する"""
        # パターン1: 直接的なパーセント値
        percent_match = re.search(r'(\d+)%', progress_text)
        if percent_match:
            return int(percent_match.group(1))
            
        # パターン2: N/M 形式（例: 45/100）
        ratio_match = re.search(r'(\d+)/(\d+)', progress_text)
        if ratio_match:
            current, total = int(ratio_match.group(1)), int(ratio_match.group(2))
            if total > 0:
                return int(current / total * 100)
                
        # パターン3: tqdmの進捗バー中のパーセント値
        tqdm_match = re.search(r'(\d+)%\|', progress_text)
        if tqdm_match:
            return int(tqdm_match.group(1))

        # パターン4: epoch [N/M] 形式
        epoch_match = re.search(r'epoch \[(\d+)/(\d+)\]', progress_text, re.IGNORECASE)
        if epoch_match:
            current, total = int(epoch_match.group(1)), int(epoch_match.group(2))
            if total > 0:
                return int(current / total * 100)
        
        # パターン5: Progress: N% 形式
        progress_match = re.search(r'progress:?\s*(\d+)%', progress_text, re.IGNORECASE)
        if progress_match:
            return int(progress_match.group(1))
            
        # パターン6: N out of M 形式
        out_of_match = re.search(r'(\d+)\s+out\s+of\s+(\d+)', progress_text, re.IGNORECASE)
        if out_of_match:
            current, total = int(out_of_match.group(1)), int(out_of_match.group(2))
            if total > 0:
                return int(current / total * 100)
            
        return None

    def debug_log(self, message):
        """デバッグログを出力（debug_enabledがTrueの場合のみ）"""
        if self.debug_enabled:
            print(f"[DEBUG] {message}")

    def notify_abnormal_termination(self, process_id, return_code):
        """プロセスの異常終了をユーザーに通知する"""
        process_info = self.processes[process_id]
        
        # 通知メッセージの作成
        process_name = process_info["name"]
        file_name = os.path.basename(process_info["file"])
        
        # 通知ウィンドウを表示（非ブロッキング）
        self.root.after(100, lambda: self.show_notification(
            f"プロセス異常終了の通知",
            f"プロセス「{process_name}」({file_name})が異常終了しました。\n"
            f"終了コード: {return_code}\n\n"
            f"詳細は出力ログを確認してください。"
        ))

        # Treeviewの該当行を赤色にハイライト
        self.process_tree.tag_configure("error", background="#FFCCCC")
        self.process_tree.item(str(process_id), tags=("error",))
    
    def show_notification(self, title, message):
        """通知ダイアログを表示する"""
        notification = tk.Toplevel(self.root)
        notification.title(title)
        notification.geometry("400x200")
        notification.transient(self.root)
        notification.resizable(False, False)
        
        # アイコンの設定
        try:
            # ウィンドウマネージャに依存
            notification.iconbitmap(default="error")
        except:
            pass
        
        # フレーム
        frame = ttk.Frame(notification, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 警告アイコン
        warning_label = ttk.Label(frame, text="⚠️", font=("Arial", 24))
        warning_label.pack(pady=(0, 10))
        
        # メッセージ
        message_label = ttk.Label(frame, text=message, wraplength=350, justify="center")
        message_label.pack(pady=10)
        
        # OKボタン
        ok_button = ttk.Button(frame, text="OK", command=notification.destroy)
        ok_button.pack(pady=10)
        
        # ウィンドウの中央配置
        notification.update_idletasks()
        width = notification.winfo_width()
        height = notification.winfo_height()
        x = (notification.winfo_screenwidth() // 2) - (width // 2)
        y = (notification.winfo_screenheight() // 2) - (height // 2)
        notification.geometry(f"{width}x{height}+{x}+{y}")
        
        # 通知音を鳴らす（可能であれば）
        try:
            notification.bell()
        except:
            pass

    # 追加: Treeviewのソート機能
    def sort_treeview(self, column, data_type, toggle_direction=True):
        """Treeviewの列に基づいてソートする関数
        
        Args:
            column (str): ソートする列名
            data_type (type): ソートするデータの型（int、strなど）
            toggle_direction (bool): ソート方向を切り替えるかどうか
        """
        if column == "#0":  # ツリーの最初の列（インデックス列）はソートしない
            return
            
        # ソート方向の切り替え
        if toggle_direction:
            self.sort_reverse = not getattr(self, 'sort_reverse', False)
        
        # 現在選択されている項目があれば、そのIDを記憶
        current_selection = self.process_tree.selection()
        
        # ソート中の列を記録
        self.sort_column = column
        
        # 並び替えるアイテムのリストを取得
        l = [(self.process_tree.set(k, column), k) for k in self.process_tree.get_children('')]
        
        try:
            # データ型に応じてソート
            if data_type == int:
                # 数値として変換可能な場合は数値でソート
                l.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=self.sort_reverse)
            else:
                # 文字列としてソート
                l.sort(reverse=self.sort_reverse)
                
            # ソート後の順序でツリービューを再構築
            for index, (_, k) in enumerate(l):
                self.process_tree.move(k, '', index)
                
            # ソートされた列のヘッダーに矢印を表示
            for col in self.process_tree["columns"]:
                if col == column:
                    # ソート方向に応じた矢印を表示
                    self.process_tree.heading(col, text=self.process_tree.heading(col)["text"].replace(" ▲", "").replace(" ▼", "") + (' ▼' if self.sort_reverse else ' ▲'))
                else:
                    # 他の列は通常表示
                    self.process_tree.heading(col, text=self.process_tree.heading(col)["text"].replace(" ▲", "").replace(" ▼", ""))
            
            # 以前の選択状態を復元
            if current_selection:
                for item_id in current_selection:
                    self.process_tree.selection_add(item_id)
                
        except Exception as e:
            print(f"ソートエラー: {e}")

    def remove_selected_processes(self):
        """選択されたプロセスをリストから削除する"""
        # 選択されたアイテムのIDを取得
        selected_items = self.process_tree.selection()
        if not selected_items:
            messagebox.showinfo("情報", "削除するプロセスを選択してください。")
            return
        
        # 削除確認
        if not messagebox.askyesno("確認", f"選択された {len(selected_items)} 個のプロセスをリストから削除しますか？"):
            return
        
        # 選択されたプロセスをリストから削除
        for item_id in selected_items:
            process_id = int(item_id)
            
            # プロセスが実行中の場合は警告
            if process_id in self.processes and self.processes[process_id]["process"].poll() is None:
                if not messagebox.askyesno("警告", f"プロセス '{self.processes[process_id]['name']}' は実行中です。\n本当に削除しますか？"):
                    continue
                
                # プロセスを強制終了
                try:
                    self.processes[process_id]["process"].terminate()
                    try:
                        self.processes[process_id]["process"].wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.processes[process_id]["process"].kill()
                except:
                    pass  # プロセスの終了に失敗しても続行
            
            # Treeviewから削除
            self.process_tree.delete(item_id)
            
            # プロセス辞書から削除（存在する場合のみ）
            if process_id in self.processes:
                del self.processes[process_id]
        
        # 選択プロセスの情報表示を更新
        self.update_progress_display()

def main():
    root = tk.Tk()
    app = ProcessControlTerminal(root)
    root.mainloop()

if __name__ == "__main__":
    main()
