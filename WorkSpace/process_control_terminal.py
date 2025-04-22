#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
import threading
import queue
import platform
import re
import traceback
import paramiko  # SSHクライアント用ライブラリ
import io
import subprocess
import time
from datetime import datetime
import pickle
import signal
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class ProcessControlTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("プロセス制御ターミナル")
        
        # ウィンドウのサイズを設定し、最小サイズも指定する
        self.root.geometry("1000x800")
        self.root.minsize(1600, 600)
        
        # 終了時のイベントを設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # インスタンス変数の初期化
        self.process_counter = 1  # プロセスのカウンター
        self.processes = {}  # プロセス情報を格納する辞書
        self.max_output_lines = 1000  # バッファの最大行数
        self.last_directory = os.path.dirname(os.path.abspath(__file__))  # 初期ディレクトリ
        self.update_timer = None  # 更新タイマー
        self.sort_column = None  # ソート列
        self.sort_reverse = False  # ソート方向
        self.process_notes = {}  # プロセスのメモを格納する辞書
        
        # SSH関連の変数
        self.ssh_client = None
        self.is_ssh_connected = False
        
        # tkinter変数の初期化
        self.file_path_var = tk.StringVar()
        self.args_var = tk.StringVar()
        self.dir_path_var = tk.StringVar(value=self.last_directory)
        self.process_name_var = tk.StringVar()
        self.confirm_var = tk.BooleanVar(value=True)  # デフォルトで確認する
        self.run_location_var = tk.StringVar(value="local")  # デフォルトはローカル実行
        self.progress_info_var = tk.StringVar(value="プロセスを選択してください")
        self.progress_detail_var = tk.StringVar(value="")
        self.memo_var = tk.StringVar()  # メモ入力用
        
        # SSH設定変数
        self.ssh_hostname_var = tk.StringVar()
        self.ssh_username_var = tk.StringVar()
        self.ssh_password_var = tk.StringVar()
        self.ssh_remote_path_var = tk.StringVar(value="~/")
        self.ssh_status_var = tk.StringVar(value="未接続")
        self.ssh_venv_name_var = tk.StringVar(value="venv")  # 仮想環境名の初期値
        
        # 暗号化キーの初期化
        self.initialize_encryption_key()
        
        # GUIを作成
        self.create_gui()
        
        # ファイル一覧を更新
        self.refresh_file_list()
        
        # Treeviewの列ヘッダーの初期ソート設定
        self.sort_column = "id"
        self.sort_reverse = False
        
        # 設定を読み込む
        self.load_config()
        
        # 保存されていたプロセス情報を復元
        self.restore_saved_processes()
        
        # 進捗表示の更新タイマーを開始
        self.start_progress_update_timer()
    
    def initialize_encryption_key(self):
        """暗号化キーを初期化またはロードする"""
        key_file = "config_key.key"
        if os.path.exists(key_file):
            # キーファイルが存在する場合はロード
            with open(key_file, "rb") as f:
                self.encryption_key = f.read()
        else:
            # 新しいキーを生成して保存
            self.encryption_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.encryption_key)
        
        # Fernetインスタンスを作成
        self.fernet = Fernet(self.encryption_key)
    
    def encrypt_password(self, password):
        """パスワードを暗号化する"""
        if not password:
            return ""
        return self.fernet.encrypt(password.encode()).decode()
    
    def decrypt_password(self, encrypted_password):
        """暗号化されたパスワードを復号化する"""
        if not encrypted_password:
            return ""
        try:
            return self.fernet.decrypt(encrypted_password.encode()).decode()
        except Exception as e:
            self.debug_log(f"パスワードの復号化に失敗しました: {e}")
            return ""
    
    def create_gui(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左右2段組みのレイアウト
        # 左側フレーム（コントロールパネル）
        left_frame = ttk.Frame(main_frame, width=500)  # 左側の幅を固定
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        left_frame.pack_propagate(False)  # サイズを固定
        
        # 右側フレーム（プロセスリストとメモ）
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # コントロールパネル
        control_panel = ttk.Frame(left_frame)
        control_panel.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # SSH接続フレーム
        ssh_frame = ttk.LabelFrame(control_panel, text="SSH接続")
        ssh_frame.pack(fill=tk.X, pady=5)
        
        # SSH設定の入力行
        ssh_settings_frame = ttk.Frame(ssh_frame)
        ssh_settings_frame.pack(fill=tk.X, pady=5)
        
        # グリッドを使用してきれいに配置
        # ホスト名
        ttk.Label(ssh_settings_frame, text="ホスト名:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ssh_settings_frame, textvariable=self.ssh_hostname_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # ユーザー名
        ttk.Label(ssh_settings_frame, text="ユーザー名:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ssh_settings_frame, textvariable=self.ssh_username_var, width=15).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # パスワード
        ttk.Label(ssh_settings_frame, text="パスワード:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ssh_settings_frame, textvariable=self.ssh_password_var, show="*", width=15).grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        
        # リモートパス
        ttk.Label(ssh_settings_frame, text="リモートパス:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ssh_settings_frame, textvariable=self.ssh_remote_path_var, width=30).grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        
        # 仮想環境名
        ttk.Label(ssh_settings_frame, text="Venv名:").grid(row=1, column=4, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(ssh_settings_frame, textvariable=self.ssh_venv_name_var, width=15).grid(row=1, column=5, sticky=tk.W, padx=5, pady=2)
        
        # 接続ステータス
        ttk.Label(ssh_settings_frame, text="状態:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(ssh_settings_frame, textvariable=self.ssh_status_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 接続/切断ボタン
        ssh_button_frame = ttk.Frame(ssh_frame)
        ssh_button_frame.pack(fill=tk.X, pady=5)
        
        # 接続・切断ボタンの作成とインスタンス変数に保存
        self.connect_button = ttk.Button(ssh_button_frame, text="接続", command=self.connect_ssh)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_button = ttk.Button(ssh_button_frame, text="切断", command=self.disconnect_ssh, state=tk.DISABLED)
        self.disconnect_button.pack(side=tk.LEFT, padx=5)
        
        # 作業ディレクトリ選択フレーム
        dir_frame = ttk.LabelFrame(control_panel, text="作業ディレクトリ", padding="5 5 5 5")
        dir_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(dir_frame, text="作業ディレクトリ:").pack(side=tk.LEFT)
        
        ttk.Entry(dir_frame, textvariable=self.dir_path_var, width=30).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(dir_frame, text="参照...", command=self.browse_directory).pack(side=tk.LEFT)
        
        # ファイル一覧フレーム
        files_frame = ttk.LabelFrame(control_panel, text="ファイル一覧", padding="5")
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
        file_frame = ttk.Frame(control_panel)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Pythonファイル:").pack(side=tk.LEFT)
        
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=30).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(file_frame, text="参照...", command=self.browse_file).pack(side=tk.LEFT)
        
        # コマンドライン引数
        args_frame = ttk.Frame(control_panel)
        args_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(args_frame, text="引数:").pack(side=tk.LEFT)
        
        ttk.Entry(args_frame, textvariable=self.args_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # プロセス名
        name_frame = ttk.Frame(control_panel)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="プロセス名:").pack(side=tk.LEFT)
        
        ttk.Entry(name_frame, textvariable=self.process_name_var, width=30).pack(side=tk.LEFT, padx=5)
        
        # メモ入力欄（追加）
        memo_frame = ttk.LabelFrame(control_panel, text="プロセスメモ", padding="5 5 5 0")
        memo_frame.pack(fill=tk.X, expand=False, pady=(5, 0))
        
        ttk.Entry(memo_frame, textvariable=self.memo_var, width=40).pack(fill=tk.X, padx=5, pady=(5, 0))
        
        # 確認ダイアログを表示するかどうか
        confirm_frame = ttk.Frame(control_panel)
        confirm_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Checkbutton(confirm_frame, text="実行前に確認ダイアログを表示", variable=self.confirm_var).pack(anchor=tk.W)
        
        # 実行場所選択フレーム
        location_frame = ttk.Frame(control_panel)
        location_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(location_frame, text="実行場所:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(location_frame, text="ローカル", variable=self.run_location_var, value="local").pack(side=tk.LEFT)
        self.remote_radio = ttk.Radiobutton(location_frame, text="SSH接続先", variable=self.run_location_var, value="remote")
        self.remote_radio.pack(side=tk.LEFT)
        
        # SSH接続がない場合はリモート実行を無効化
        if not self.is_ssh_connected:
            self.remote_radio.config(state=tk.DISABLED)
        
        # 実行ボタンのスタイル設定
        style = ttk.Style()
        style.configure('Big.TButton', font=('Helvetica', 12, 'bold'), padding=(10, 10))
        
        # 実行ボタン
        exec_button = ttk.Button(control_panel, text="実行", command=self.start_process, style='Big.TButton')
        exec_button.pack(fill=tk.X, pady=15, ipady=8)  # ipadyで内部パディングを増やして高さを拡大
        
        # プロセスリスト（右側フレームに配置）
        list_frame = ttk.LabelFrame(right_frame, text="実行中プロセス", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # 進捗バーインラインフレーム
        self.progress_frames = {}
        
        # Treeviewの列を定義（列の順序を変更）
        columns = ("id", "file", "args", "status", "progress_bar", "location", "progress_text", "start_time", "pid", "memo")
        self.process_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=8, selectmode="extended")
        
        # 各列の見出しを設定（見出しのラベルを変更）
        self.process_tree.heading("id", text="ID", command=lambda: self.sort_treeview("id", int))
        self.process_tree.heading("file", text="ファイル", command=lambda: self.sort_treeview("file", str))
        self.process_tree.heading("args", text="引数", command=lambda: self.sort_treeview("args", str))
        self.process_tree.heading("status", text="状態", command=lambda: self.sort_treeview("status", str))
        self.process_tree.heading("progress_bar", text="進捗バー")  # 進捗バーはソート対象外
        self.process_tree.heading("location", text="実行場所", command=lambda: self.sort_treeview("location", str))
        self.process_tree.heading("progress_text", text="進捗詳細", command=lambda: self.sort_treeview("progress_text", str))
        self.process_tree.heading("start_time", text="開始時間", command=lambda: self.sort_treeview("start_time", str))
        self.process_tree.heading("pid", text="PID", command=lambda: self.sort_treeview("pid", int))
        self.process_tree.heading("memo", text="メモ", command=lambda: self.sort_treeview("memo", str))
        
        # 列の幅を設定
        self.process_tree.column("id", width=20)
        self.process_tree.column("file", width=150)
        self.process_tree.column("args", width=80)
        self.process_tree.column("status", width=70)
        self.process_tree.column("progress_bar", width=250)
        self.process_tree.column("location", width=70)
        self.process_tree.column("progress_text", width=120)
        self.process_tree.column("start_time", width=140)
        self.process_tree.column("pid", width=60)
        self.process_tree.column("memo", width=100)
        
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
    
    def execute_remote_process(self, file_path, args=None):
        """SSHを使用してリモートでプロセスを実行する"""
        if not self.is_ssh_connected or not self.ssh_client:
            raise Exception("SSH接続がありません。")
        
        if args is None:
            args = []
            
        # ファイル名を取得
        file_name = os.path.basename(file_path)
        
        # ローカルファイルが存在するか確認
        if not os.path.isfile(file_path):
            raise Exception(f"ファイルが存在しません: {file_path}")
        
        # リモートのターゲットディレクトリを取得
        remote_dir = self.ssh_remote_path_var.get().strip()
        if not remote_dir:
            remote_dir = "/tmp"  # デフォルトディレクトリ
        
        # リモートディレクトリの存在確認
        try:
            check_dir_cmd = f"if [ -d \"{remote_dir}\" ]; then echo 'exists'; else echo 'not_exists'; fi"
            stdin, stdout, stderr = self.ssh_client.exec_command(check_dir_cmd)
            result = stdout.read().decode().strip()
            
            if result == 'not_exists':
                # ディレクトリが存在しない場合は作成を試みる
                create_dir_cmd = f"mkdir -p \"{remote_dir}\""
                stdin, stdout, stderr = self.ssh_client.exec_command(create_dir_cmd)
                err = stderr.read().decode().strip()
                if err:
                    # ディレクトリ作成に失敗した場合は/tmpを使用
                    messagebox.showwarning("警告", f"リモートディレクトリ '{remote_dir}' の作成に失敗しました。/tmpを使用します。")
                    remote_dir = "/tmp"
        except Exception as e:
            messagebox.showwarning("警告", f"リモートディレクトリの確認中にエラーが発生しました: {e}")
            remote_dir = "/tmp"  # エラーが発生した場合はデフォルトに戻す
        
        # リモートのファイルパスを設定
        remote_file = f"{remote_dir}/{file_name}"
        
        # ファイルをアップロード
        # Windows環境のエスケープシーケンスの問題に対処
        upload_success = False
        
        # 方法1: 内容を読み込んでエスケープシーケンスを修正してからアップロード
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Windowsパスのバックスラッシュ問題を修正
            file_content = file_content.replace('\\', '\\\\')
            
            # 一時ファイルに保存
            temp_file_path = os.path.join(os.path.dirname(file_path), f"_temp_{file_name}")
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
                
            # 修正したファイルをアップロード
            sftp = self.ssh_client.open_sftp()
            try:
                sftp.put(temp_file_path, remote_file)
                sftp.chmod(remote_file, 0o755)  # 実行権限を付与
                upload_success = True
                self.debug_log("UTF-8エンコーディングでファイルを修正してアップロードしました")
            finally:
                sftp.close()
                
            # 一時ファイルを削除
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
        except UnicodeDecodeError:
            self.debug_log("UTF-8でのデコードに失敗しました。バイナリモードでアップロードを試みます。")
        except Exception as e:
            self.debug_log(f"ファイル修正アップロード中にエラー: {e}")
        
        # 方法1が失敗した場合、通常の方法でアップロード
        if not upload_success:
            try:
                sftp = self.ssh_client.open_sftp()
                try:
                    sftp.put(file_path, remote_file)
                    sftp.chmod(remote_file, 0o755)  # 実行権限を付与
                    upload_success = True
                    self.debug_log("通常方法でファイルをアップロードしました")
                finally:
                    sftp.close()
            except Exception as e:
                self.debug_log(f"通常アップロード中にエラー: {e}")
                raise Exception(f"ファイルのアップロードに失敗しました: {e}")
        
        # venvの検出
        venv_activate = ""
        try:
            # 指定されたvenv名を取得
            venv_name = self.ssh_venv_name_var.get().strip()
            if not venv_name:
                venv_name = "venv"  # デフォルト値
                
            # 優先して検索するパス（設定で指定されたvenv名を使用）
            priority_venv_paths = [
                f"{remote_dir}/{venv_name}/bin/activate",
                f"{remote_dir}/.{venv_name}/bin/activate",
                f"{os.path.dirname(remote_dir)}/{venv_name}/bin/activate",
                f"{os.path.dirname(remote_dir)}/.{venv_name}/bin/activate"
            ]
            
            # リモートのホームディレクトリを取得
            home_cmd = "echo $HOME"
            stdin, stdout, stderr = self.ssh_client.exec_command(home_cmd)
            home_dir = stdout.read().decode().strip()
            
            # ホームディレクトリの優先venvパスも追加
            if home_dir:
                priority_venv_paths.extend([
                    f"{home_dir}/{venv_name}/bin/activate",
                    f"{home_dir}/.{venv_name}/bin/activate"
                ])
            
            # 標準的なvenvパス（デフォルトのvenv名）
            standard_venv_paths = [
                f"{remote_dir}/venv/bin/activate", 
                f"{remote_dir}/.venv/bin/activate",
                f"{os.path.dirname(remote_dir)}/venv/bin/activate",
                f"{os.path.dirname(remote_dir)}/.venv/bin/activate"
            ]
            
            # ホームディレクトリの標準的なvenvパスも追加
            if home_dir:
                standard_venv_paths.extend([
                    f"{home_dir}/venv/bin/activate",
                    f"{home_dir}/.venv/bin/activate"
                ])
            
            # 優先パスを先に検索
            self.debug_log(f"優先venv検索: {priority_venv_paths}")
            for venv_path in priority_venv_paths:
                check_venv_cmd = f"if [ -f \"{venv_path}\" ]; then echo 'exists'; else echo 'not_exists'; fi"
                stdin, stdout, stderr = self.ssh_client.exec_command(check_venv_cmd)
                result = stdout.read().decode().strip()
                if result == 'exists':
                    venv_activate = f"source \"{venv_path}\" && "
                    self.debug_log(f"優先リモート仮想環境が見つかりました: {venv_path}")
                    messagebox.showinfo("仮想環境", f"指定された仮想環境を使用します: {venv_path}")
                    break
            
            # 優先パスで見つからなかった場合、標準パスを検索
            if not venv_activate:
                self.debug_log("標準venv検索を開始します")
                for venv_path in standard_venv_paths:
                    check_venv_cmd = f"if [ -f \"{venv_path}\" ]; then echo 'exists'; else echo 'not_exists'; fi"
                    stdin, stdout, stderr = self.ssh_client.exec_command(check_venv_cmd)
                    result = stdout.read().decode().strip()
                    if result == 'exists':
                        venv_activate = f"source \"{venv_path}\" && "
                        self.debug_log(f"標準リモート仮想環境が見つかりました: {venv_path}")
                        messagebox.showinfo("仮想環境", f"標準の仮想環境を使用します: {venv_path}")
                        break
        except Exception as e:
            self.debug_log(f"venv検出エラー: {e}")
        
        # コマンドを構築
        python_cmd = "python3"
        
        # venvがあればその環境のpythonを使用
        if venv_activate:
            python_cmd_check = f"cd {remote_dir} && {venv_activate} which python"
            stdin, stdout, stderr = self.ssh_client.exec_command(python_cmd_check)
            if stdout.channel.recv_exit_status() == 0:
                python_path = stdout.read().decode().strip()
                if python_path:
                    python_cmd = python_path
                    self.debug_log(f"仮想環境のPythonを使用: {python_cmd}")
        
        cmd = f"cd {remote_dir} && {venv_activate}{python_cmd} {file_name}"
        if args:
            cmd += f" {' '.join(args)}"
        
        # バックグラウンドでプロセスを実行し、出力をファイルにリダイレクト
        output_file = f"/tmp/{file_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
        
        # 確実に終了コードをログに記録するよう修正したコマンド
        if venv_activate:
            # venv_activateから実際のパスを抽出
            venv_path_match = re.search(r'source "([^"]+)"', venv_activate)
            if venv_path_match:
                actual_venv_path = venv_path_match.group(1)
                # ログインシェルを使用して仮想環境を確実にアクティベート
                bg_cmd = f"""nohup bash --login -c '
cd {remote_dir}
echo "====== 仮想環境前の Python 情報 ======" >> {output_file}
which python >> {output_file} 2>&1
python --version >> {output_file} 2>&1
echo "====== 仮想環境アクティベーション開始 ======" >> {output_file}
source {actual_venv_path}
echo "====== 仮想環境後の Python 情報 ======" >> {output_file}
which python >> {output_file} 2>&1
python --version >> {output_file} 2>&1
echo "====== スクリプト実行開始 ======" >> {output_file}
{python_cmd} {file_name}"""
                
                # 引数があれば追加（それぞれの引数をエスケープして追加）
                if args:
                    escaped_args = []
                    for arg in args:
                        # 引数内のシングルクォートをエスケープ
                        escaped_arg = arg.replace("'", "'\\''")
                        escaped_args.append(f"'{escaped_arg}'")
                    bg_cmd += f" {' '.join(escaped_args)}"
                
                # 終了コードを記録してリダイレクト（確実に記録するよう修正）
                bg_cmd += f"""
EXIT_CODE=$?
echo ""
echo "プロセスが終了しました。終了コード: $EXIT_CODE" >> {output_file}
exit $EXIT_CODE' > {output_file} 2>&1 & echo $!"""
                self.debug_log(f"仮想環境を使用: {actual_venv_path}")
            else:
                # 抽出に失敗した場合のフォールバック
                bg_cmd = f"""nohup bash -c '
cd {remote_dir}
echo "====== venv コマンド実行 ======" >> {output_file}
{venv_activate} {python_cmd} {file_name}"""
                if args:
                    escaped_args = []
                    for arg in args:
                        escaped_arg = arg.replace("'", "'\\''")
                        escaped_args.append(f"'{escaped_arg}'")
                    bg_cmd += f" {' '.join(escaped_args)}"
                bg_cmd += f"""
EXIT_CODE=$?
echo ""
echo "プロセスが終了しました。終了コード: $EXIT_CODE" >> {output_file}
exit $EXIT_CODE' > {output_file} 2>&1 & echo $!"""
        else:
            # 仮想環境なしの場合、直接実行するシンプルなコマンド
            python_cmd = "python3"  # デフォルトのPythonコマンド
            
            # 使用可能なPythonコマンドを確認
            check_python_cmd = "which python3 || which python || echo 'not_found'"
            stdin, stdout, stderr = self.ssh_client.exec_command(check_python_cmd)
            python_path = stdout.read().decode().strip()
            
            if python_path == 'not_found':
                raise Exception("リモートサーバーにPythonが見つかりません。Pythonをインストールしてください。")
            
            # 有効なPythonパスが見つかった場合は使用
            if python_path != 'not_found':
                python_cmd = python_path
            
            # シンプルにbashを使って実行（ログイン環境不要）
            bg_cmd = f"""nohup bash -c '
cd {remote_dir}
echo "====== Python情報 ======" >> {output_file}
{python_cmd} --version >> {output_file} 2>&1
echo "====== スクリプト実行開始 ======" >> {output_file}
{python_cmd} {file_name}"""
            
            # 引数があれば追加
            if args:
                escaped_args = []
                for arg in args:
                    escaped_arg = arg.replace("'", "'\\''")
                    escaped_args.append(f"'{escaped_arg}'")
                bg_cmd += f" {' '.join(escaped_args)}"
            
            # 終了コードを記録（改行を追加して確実に記録）
            bg_cmd += f"""
EXIT_CODE=$?
echo ""
echo "プロセスが終了しました。終了コード: $EXIT_CODE" >> {output_file}
exit $EXIT_CODE' > {output_file} 2>&1 & echo $!"""
        
        # 実行コマンドをデバッグログに出力
        self.debug_log(f"リモートコマンド実行: {bg_cmd}")
        
        # コマンドを実行
        stdin, stdout, stderr = self.ssh_client.exec_command(bg_cmd)
        pid = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        
        if err:
            self.debug_log(f"リモートコマンド実行エラー: {err}")
            # エラーがあってもPIDが取得できていれば処理を続行
            if not pid:
                raise Exception(f"リモートプロセスの実行に失敗しました: {err}")
        
        # リモートプロセスを表すオブジェクトを生成
        class RemoteProcess:
            def __init__(self, pid, output_file):
                self.pid = int(pid)
                self.output_file = output_file
                self._returncode = None
                
            def poll(self):
                return self._returncode
                
            def wait(self, timeout=None):
                return self._returncode
                
            def terminate(self):
                # 実際の終了処理はProcessControlTerminalクラスのメソッドで行う
                pass
                
            def kill(self):
                # 実際の終了処理はProcessControlTerminalクラスのメソッドで行う
                pass
        
        # リモートプロセスオブジェクトを生成
        process = RemoteProcess(pid, output_file)
        
        # 出力ファイル読み取りを開始
        self.root.after(1000, self.read_remote_output, self.process_counter, output_file)
        
        return process, output_file

    def monitor_process(self, process_id):
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        process = process_info["process"]
        
        # プロセスの原プロセスIDを保存（プロセスが置き換えられた場合に備えて）
        original_pid = process.pid
        
        # リモート実行かどうかを確認
        location = process_info.get("location", "local")
        
        if location == "remote":
            # リモートプロセスの場合
            # 最初の状態確認は少し待ってから行う
            self.root.after(5000, self.check_remote_process_periodically, process_id, original_pid)
        else:
            # ローカルプロセスの場合
            # プロセスが終了するまで待機
            process.wait()
            
            # プロセスが既に再起動などで置き換えられていないか確認
            if process_id in self.processes and self.processes[process_id]["process"].pid == original_pid:
                # GUIの更新はメインスレッドで行う
                self.root.after(100, self.update_process_status, process_id)
    
    def check_remote_process_periodically(self, process_id, original_pid):
        """リモートプロセスの状態を定期的に確認する"""
        if process_id not in self.processes:
            return
            
        # プロセスが置き換えられていないか確認
        if self.processes[process_id]["process"].pid != original_pid:
            return  # プロセスが置き換えられている場合は終了
            
        process_info = self.processes[process_id]
        process = process_info["process"]
        
        # プロセスのステータスを確認
        return_code = self.check_remote_process_status(process.pid)
        
        # プロセスの終了コードを設定
        if return_code is not None:
            # 終了コードを設定
            process._returncode = return_code
            # GUIの更新
            self.root.after(100, self.update_process_status, process_id)
        else:
            # まだ実行中なので、再度スケジュール
            self.root.after(5000, self.check_remote_process_periodically, process_id, original_pid)

    def update_process_status(self, process_id):
        if process_id not in self.processes:
            return
            
        process_info = self.processes[process_id]
        process = process_info["process"]
        
        # 終了コードを取得
        return_code = process.poll()
        
        # リモート実行かどうかを確認
        location = process_info.get("location", "local")
        location_display = "ローカル" if location == "local" else "リモート"
        
        # デバッグログ
        self.debug_log(f"プロセスID {process_id} (PID {process.pid}) のステータス更新: " + 
                      f"場所={location}, 終了コード={return_code}")
        
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
                location_display,  # 実行場所
                process_info.get("progress_text", ""),  # 進捗テキスト
                process_info["start_time"],
                process.pid,
                process_info["memo"]
            )
        )
        
        # 異常終了の場合のみ通知（手動停止でない場合）
        if notify:
            self.notify_abnormal_termination(process_id, return_code)
    
    def check_remote_process_status(self, pid):
        """リモートプロセスのステータスを確認する"""
        if not self.is_ssh_connected or not self.ssh_client:
            return 1  # SSH接続がない場合はエラーとして扱う
            
        try:
            # プロセスの存在を確認するコマンド
            check_cmd = f"ps -p {pid} > /dev/null 2>&1 && echo 'running' || echo $?"
            stdin, stdout, stderr = self.ssh_client.exec_command(check_cmd)
            result = stdout.read().decode().strip()
            
            self.debug_log(f"リモートプロセス[{pid}]の状態確認: {result}")
            
            if result == 'running':
                return None  # プロセスが実行中
            else:
                # 終了コードの取得（ps -p の終了コードを優先する）
                try:
                    # 数値に変換できる場合は終了コード
                    return_code = int(result)
                    # 通常はpsコマンドの終了コード1（プロセスが存在しない）が返るが、
                    # この段階では実際のプロセスの終了コードはまだわからない
                    
                    # ログファイルから終了メッセージを探す試み
                    for process_id, process_info in self.processes.items():
                        if hasattr(process_info["process"], 'pid') and process_info["process"].pid == int(pid):
                            if "output_file" in process_info["process"].__dict__:
                                output_file = process_info["process"].output_file
                                # 終了コードをログから検索 (最後の行を優先)
                                search_cmd = f"grep -a '終了コード: ' {output_file} | tail -1"
                                stdin, stdout, stderr = self.ssh_client.exec_command(search_cmd)
                                exit_msg = stdout.read().decode().strip()
                                self.debug_log(f"終了コード検索結果: {exit_msg}")
                                
                                if exit_msg:
                                    match = re.search(r'終了コード: (\d+)', exit_msg)
                                    if match:
                                        real_code = int(match.group(1))
                                        self.debug_log(f"検出された実際の終了コード: {real_code}")
                                        return real_code
                            break
                    
                    # 終了コードが検出できなかった場合はpsコマンドの終了コードを返す
                    self.debug_log(f"ログからの終了コード検出失敗。PSコマンドの返り値を使用: {return_code}")
                    return return_code
                except ValueError:
                    # 検出できなかった場合は正常終了と見なす
                    self.debug_log("数値変換エラー。終了コード0を返します。")
                    return 0
        except Exception as e:
            self.debug_log(f"リモートプロセス状態確認エラー: {e}")
            return 1  # エラーが発生した場合はエラーコードを返す
    
    def terminate_remote_process(self, pid, remote_dir, file_name):
        """リモートプロセスを終了する"""
        if not self.is_ssh_connected or not self.ssh_client:
            return
            
        try:
            # プロセスを強制終了
            kill_cmd = f"kill -9 {pid}"
            self.ssh_client.exec_command(kill_cmd)
            
            # 一時ファイルを削除（オプション）
            # cleanup_cmd = f"rm -f {remote_dir}/{file_name} /tmp/{file_name}_*.log"
            # self.ssh_client.exec_command(cleanup_cmd)
        except Exception as e:
            print(f"リモートプロセス終了エラー: {e}")
    
    def read_remote_output(self, process_id, output_file):
        """リモートプロセスの出力ファイルを定期的に読み取る"""
        if process_id not in self.processes:
            return
            
        try:
            # 出力ファイルを読み取るコマンド
            cmd = f"cat {output_file} 2>/dev/null || echo ''"
            result = self.execute_remote_command(cmd)
            output = result.get("stdout", "")
            
            # 既存の出力と比較
            current_output = self.processes[process_id].get("raw_output", "")
            
            # 新しい出力がある場合のみ処理
            if output and output != current_output:
                # 差分を取得
                new_lines = output[len(current_output):] if len(current_output) < len(output) else output
                
                # 差分を行ごとに分割して追加
                for line in new_lines.splitlines(True):  # Trueで改行を保持
                    self.processes[process_id]["output"].append(line)
                    
                    # tqdmプログレスバーの検出
                    if '%' in line or 'it/s' in line or re.search(r'\[\d+:\d+<\d+:\d+', line) or re.search(r'\d+/\d+', line):
                        progress_text = line.strip()
                        if progress_text:
                            old_progress = self.processes[process_id].get("progress", "")
                            if progress_text != old_progress:
                                self.processes[process_id]["progress"] = progress_text
                                percent = self.extract_progress_percent(progress_text)
                                if percent is not None:
                                    self.processes[process_id]["progress_percent"] = percent
                
                # 新しい出力全体を保存
                self.processes[process_id]["raw_output"] = output
                
                # バッファが大きくなりすぎないように制限
                if len(self.processes[process_id]["output"]) > self.max_output_lines:
                    self.processes[process_id]["output"] = self.processes[process_id]["output"][-self.max_output_lines:]
        except Exception as e:
            self.processes[process_id]["output"].append(f"[読み取りエラー] {str(e)}\n")
            
        # 引き続き監視
        if process_id in self.processes:
            self.root.after(1000, self.read_remote_output, process_id, output_file)
    
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
        
        # 実行場所を取得
        location = self.run_location_var.get()
        
        # SSH実行の場合、接続を確認
        if location == "remote" and not self.is_ssh_connected:
            messagebox.showerror("エラー", "SSH接続されていません。ローカル実行に切り替えます。")
            location = "local"
            self.run_location_var.set("local")
        
        # 確認ダイアログが有効ならば、実行前に確認
        if self.confirm_var.get():
            location_text = "ローカル" if location == "local" else "リモート(SSH)"
            if not messagebox.askyesno("確認", f"以下のプログラムを実行しますか？\n\nファイル: {file_path}\n引数: {' '.join(args)}\nプロセス名: {process_name}\n実行場所: {location_text}"):
                return
        
        try:
            # 実行場所に応じてプロセスを起動
            output_file = None
            if location == "remote":
                # リモート実行
                process, output_file = self.execute_remote_process(file_path, args)
            else:
                # ローカル実行
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
        
        # 実行場所の表示文字列
        location_display = "ローカル実行" if location == "local" else "SSH実行"
        
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
            "location": location,
            "manually_stopped": False,  # 手動停止フラグを追加
            "raw_output": "" if location == "remote" else None  # リモート実行用の出力バッファ
        }
        
        # メモを記録
        self.process_notes[process_id] = memo
        
        # Treeviewのアイテムとして追加（列の順序を変更）
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
                location_display,  # 実行場所
                "",  # 進捗テキスト列
                start_time,
                process.pid,
                memo
            )
        )
        
        # 現在のソート順を適用（方向は変えない）
        if hasattr(self, 'sort_column') and self.sort_column:
            self.sort_treeview(self.sort_column, int if self.sort_column in ["id", "pid"] else str, toggle_direction=False)
        
        # 実行場所に応じて出力の監視方法を変更
        if location == "remote":
            # リモート出力ファイルの読み取りを開始
            self.root.after(1000, self.read_remote_output, process_id, output_file)
        else:
            # ローカルの場合は通常の出力読み取りスレッドを開始
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
            
            # 実行場所を取得
            location = process_info.get("location", "local")
            
            # SSH実行の場合、接続を確認
            if location == "remote" and not self.is_ssh_connected:
                if len(selected_items) == 1:  # 単一選択の場合のみ個別メッセージ
                    if not messagebox.askyesno("SSH接続エラー", "SSH接続が失われています。ローカル実行に切り替えますか？"):
                        # キャンセルされた場合、処理をスキップ
                        continue
                    else:
                        location = "local"  # ローカル実行に切り替え
                else:
                    # 複数選択の場合は自動的にローカル実行に切り替え
                    location = "local"
            
            try:
                output_file = None
                # 実行場所に応じてプロセスを起動
                if location == "remote":
                    # リモート実行
                    new_process, output_file = self.execute_remote_process(process_info["file"], process_info["args"])
                else:
                    # ローカル実行
                    # 環境変数を設定（UTF-8を使用）
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    
                    new_process = subprocess.Popen(
                        [sys.executable, process_info["file"]] + process_info["args"],
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
                self.processes[process_id]["location"] = location  # 実行場所を保持
                
                # 実行場所の表示テキスト
                location_display = "ローカル実行" if location == "local" else "SSH実行"
                
                # Treeviewを更新（タグもリセット）
                self.process_tree.item(
                    str(process_id),
                    values=(
                        process_id,
                        os.path.basename(process_info["file"]),
                        " ".join(process_info["args"]) if process_info["args"] else "",
                        "実行中",
                        "",  # 進捗バー表示をリセット
                        location_display,  # 実行場所を表示
                        "",  # 進捗テキスト表示をリセット
                        now,
                        new_process.pid,
                        process_info["memo"]
                    ),
                    tags=()  # エラーを示すタグをクリア
                )
                
                # 実行場所に応じて出力の監視方法を変更
                if location == "remote":
                    # リモート出力ファイルの読み取りを開始
                    self.root.after(1000, self.read_remote_output, process_id, output_file)
                else:
                    # ローカルの場合は通常の出力読み取りスレッドを開始
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
        
        # プロセスが保存済み状態かチェック
        if process_info["status"] == "保存済み":
            messagebox.showinfo("情報", "このプロセスは保存された情報から復元されたもので、出力データはありません。")
            return
        
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
        # 現在の設定とプロセス情報を保存
        saved_processes = {}
        
        # 保存するプロセス情報をフィルタリング
        for process_id, process_info in self.processes.items():
            # プロセス情報から保存するべき情報だけを抽出
            saved_process = {
                "file": process_info["file"],
                "args": process_info["args"],
                "name": process_info.get("name", os.path.basename(process_info["file"])),
                "start_time": process_info["start_time"],
                "status": process_info["status"],
                "memo": process_info.get("memo", ""),
                "pid": process_info.get("pid", 0),
                "location": process_info.get("location", "local"),  # 実行場所を保存
                # プロセスオブジェクト自体は保存できないので除外
            }
            saved_processes[str(process_id)] = saved_process
        
        # SSH設定を保存（パスワードは暗号化して保存）
        ssh_settings = {
            "hostname": self.ssh_hostname_var.get(),
            "username": self.ssh_username_var.get(),
            "remote_path": self.ssh_remote_path_var.get(),
            "venv_name": self.ssh_venv_name_var.get()
        }
        
        # 現在のパスワードを暗号化して保存
        current_pwd = self.ssh_password_var.get()
        if current_pwd:
            ssh_settings["encrypted_password"] = self.encrypt_password(current_pwd)
        
        config = {
            "process_notes": {k: self.processes[k]["memo"] for k in self.processes if "memo" in self.processes[k]},
            "saved_processes": saved_processes,  # プロセス情報を保存
            "recent_files": list(set([p["file"] for p in self.processes.values() if isinstance(p["file"], str) and p["file"]])),
            "last_directory": self.last_directory,
            "work_directory": self.dir_path_var.get(),
            "process_counter": self.process_counter,  # プロセスカウンタも保存
            "ssh_settings": ssh_settings  # SSH設定を保存
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
                
                # SSH設定を読み込む
                if "ssh_settings" in config:
                    ssh_settings = config["ssh_settings"]
                    self.ssh_hostname_var.set(ssh_settings.get("hostname", ""))
                    self.ssh_username_var.set(ssh_settings.get("username", ""))
                    self.ssh_remote_path_var.set(ssh_settings.get("remote_path", ""))
                    self.ssh_venv_name_var.set(ssh_settings.get("venv_name", "venv"))
                    
                    # 暗号化されたパスワードがあれば復号化して設定
                    if "encrypted_password" in ssh_settings:
                        encrypted_pwd = ssh_settings["encrypted_password"]
                        decrypted_pwd = self.decrypt_password(encrypted_pwd)
                        self.ssh_password_var.set(decrypted_pwd)
                        
                        # パスワードがあれば自動接続（オプション）
                        if decrypted_pwd and ssh_settings.get("hostname") and ssh_settings.get("username"):
                            # 自動接続するかユーザーに確認
                            # self.root.after(1000, self.ask_reconnect)
                            # 確認なしで自動接続する
                            self.root.after(1000, self.auto_reconnect)
                
                # 最後に選択したディレクトリを設定
                self.last_directory = config.get("last_directory", os.path.dirname(os.path.abspath(__file__)))
                
                # 作業ディレクトリを設定
                work_dir = config.get("work_directory", os.path.dirname(os.path.abspath(__file__)))
                if os.path.isdir(work_dir):
                    self.dir_path_var.set(work_dir)
                
                # プロセスカウンタを復元
                if "process_counter" in config:
                    self.process_counter = config["process_counter"]
                    
                # 保存されていたプロセスを後からロードするための設定
                self.saved_processes_data = config.get("saved_processes", {})
                    
        except Exception as e:
            self.debug_log(f"設定の読み込みに失敗しました: {e}")
            # デフォルト設定
            self.dir_path_var.set(os.path.dirname(os.path.abspath(__file__)))
            self.ssh_remote_path_var.set("~/")
            self.saved_processes_data = {}
    
    def restore_saved_processes(self):
        """保存されていたプロセス情報を復元"""
        try:
            # 保存されていたプロセスデータを処理
            for process_id_str, process_info in self.saved_processes_data.items():
                try:
                    process_id = int(process_id_str)
                    
                    # 保存時の終了状態を確認（終了している場合は'終了'、それ以外は'中断'）
                    status = "終了" if process_info.get("status") == "終了" else "中断"
                    
                    # 実行場所
                    location = process_info.get("location", "local")
                    location_display = "ローカル実行" if location == "local" else "SSH実行"
                    
                    # Treeviewにアイテムとして追加（列の順序を変更）
                    self.process_tree.insert(
                        "",
                        "end",
                        iid=str(process_id),
                        values=(
                            process_id,
                            os.path.basename(process_info["file"]),
                            " ".join(process_info["args"]) if process_info["args"] else "",
                            status,
                            "",  # 進捗バー列
                            location_display,  # 実行場所
                            "",  # 進捗テキスト列
                            process_info["start_time"],
                            process_info.get("pid", ""),
                            process_info.get("memo", "")
                        )
                    )
                    
                    # ダミープロセスオブジェクトを作成
                    dummy_process = type('obj', (object,), {
                        'stdout': None,
                        'poll': lambda: 0,  # 終了状態を示すため0を返す
                        'terminate': lambda: None,
                        'wait': lambda: None,
                        'pid': process_info.get("pid", 0)
                    })
                    
                    self.processes[process_id] = {
                        "process": dummy_process,
                        "file": process_info["file"],
                        "args": process_info["args"],
                        "name": process_info.get("name", os.path.basename(process_info["file"])),
                        "start_time": process_info["start_time"],
                        "output": [],  # 出力は空
                        "status": status,
                        "progress": "",
                        "progress_percent": 0,
                        "memo": process_info.get("memo", ""),
                        "manually_stopped": True,  # 手動停止扱い
                        "location": location
                    }
                except Exception as e:
                    self.debug_log(f"プロセスID {process_id_str} の復元に失敗: {e}")
                    continue
            
            # 保存したデータをクリア
            self.saved_processes_data = {}
            
            # 現在のソート順を適用
            if hasattr(self, 'sort_column') and self.sort_column and self.process_tree.get_children():
                self.sort_treeview(self.sort_column, int if self.sort_column in ["id", "pid"] else str, toggle_direction=False)
            
            # 保存されたSSH情報で自動的に再接続
            self.auto_reconnect()
            
        except Exception as e:
            self.debug_log(f"保存済みプロセスの復元に失敗しました: {e}")
    
    def ask_reconnect(self):
        """保存されたSSH情報で再接続するか確認"""
        hostname = self.ssh_hostname_var.get()
        username = self.ssh_username_var.get()
        if hostname and username and messagebox.askyesno("SSH再接続", 
                                                       f"{username}@{hostname}に自動的に接続しますか？"):
            self.connect_ssh()
            
    def auto_reconnect(self):
        """保存されたSSH情報で確認なしに自動的に再接続する"""
        hostname = self.ssh_hostname_var.get()
        username = self.ssh_username_var.get()
        if hostname and username:
            self.debug_log(f"SSHに自動再接続します: {username}@{hostname}")
            self.connect_ssh(silent=True)
    
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
                            
                            # Treeviewの表示を更新（列の順序変更に合わせて修正）
                            new_values[4] = f"{progress_bar} {percent:3d}%"  # 5番目（インデックス4）が進捗バー列
                            new_values[6] = progress_text                # 7番目（インデックス6）が進捗テキスト列
                        else:
                            # 進捗情報がない場合は空のバーを表示
                            new_values[4] = "░░░░░░░░░░░░░░░░░ 0%"
                            new_values[6] = "進行状況なし"
                        
                        self.process_tree.item(str(process_id), values=new_values)
                except Exception as e:
                    self.debug_log(f"進捗表示の更新中にエラーが発生しました: {e}")
                    # 処理を続行するために例外を無視
    
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
        if hasattr(self, 'debug_enabled') and self.debug_enabled:
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

    def update_remote_execution_state(self):
        """SSH接続状態に応じてリモート実行ラジオボタンの状態を更新する"""
        if hasattr(self, "remote_radio"):
            if self.is_ssh_connected:
                self.remote_radio.config(state=tk.NORMAL)
            else:
                # SSH接続がない場合はローカル実行を強制
                self.run_location_var.set("local")
                self.remote_radio.config(state=tk.DISABLED)

    def connect_ssh(self, silent=False):
        """SSHサーバーに接続する"""
        hostname = self.ssh_hostname_var.get().strip()
        username = self.ssh_username_var.get().strip()
        password = self.ssh_password_var.get()
        remote_path = self.ssh_remote_path_var.get().strip()
        
        if not hostname or not username:
            if not silent:
                messagebox.showerror("接続エラー", "ホスト名とユーザー名は必須です。")
            return
        
        try:
            # 既存の接続を閉じる
            if self.ssh_client and self.is_ssh_connected:
                self.disconnect_ssh(silent=True)
            
            # SSH接続を作成
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.ssh_status_var.set("接続中...")
            self.root.update_idletasks()
            
            # 接続を実行
            self.ssh_client.connect(
                hostname=hostname,
                username=username,
                password=password,
                timeout=10
            )
            
            self.is_ssh_connected = True
            self.ssh_status_var.set("接続済み")
            
            # ボタンの状態を更新
            if hasattr(self, "connect_button"):
                self.connect_button.config(state=tk.DISABLED)
            if hasattr(self, "disconnect_button"):
                self.disconnect_button.config(state=tk.NORMAL)
            
            # リモート実行ラジオボタンの状態を更新
            self.update_remote_execution_state()
            
            # リモートディレクトリの設定
            if remote_path:
                # まずディレクトリが存在するか確認
                check_dir_cmd = f"if [ -d \"{remote_path}\" ]; then echo 'exists'; else echo 'not_exists'; fi"
                stdin, stdout, stderr = self.ssh_client.exec_command(check_dir_cmd)
                result = stdout.read().decode().strip()
                
                if result == 'not_exists':
                    # ディレクトリが存在しない場合は作成
                    if silent:
                        # サイレントモードでは自動的に作成
                        create_dir_cmd = f"mkdir -p \"{remote_path}\""
                        stdin, stdout, stderr = self.ssh_client.exec_command(create_dir_cmd)
                        err = stderr.read().decode().strip()
                        if err:
                            # ホームディレクトリを代わりに使用
                            stdin, stdout, stderr = self.ssh_client.exec_command("echo $HOME")
                            home_dir = stdout.read().decode().strip()
                            self.ssh_remote_path_var.set(home_dir)
                            remote_path = home_dir
                    else:
                        create_dir_msg = messagebox.askyesno("確認", f"リモートディレクトリ '{remote_path}' が存在しません。作成しますか？")
                        if create_dir_msg:
                            create_dir_cmd = f"mkdir -p \"{remote_path}\""
                            stdin, stdout, stderr = self.ssh_client.exec_command(create_dir_cmd)
                            err = stderr.read().decode().strip()
                            if err:
                                messagebox.showwarning("警告", f"ディレクトリの作成に失敗しました: {err}")
                                # ホームディレクトリを代わりに使用
                                stdin, stdout, stderr = self.ssh_client.exec_command("echo $HOME")
                                home_dir = stdout.read().decode().strip()
                                self.ssh_remote_path_var.set(home_dir)
                                remote_path = home_dir
                
                # パスの確認 (作成成功後またはすでに存在する場合)
                stdin, stdout, stderr = self.ssh_client.exec_command(f"cd {remote_path} && pwd")
                if stdout.channel.recv_exit_status() == 0:
                    actual_path = stdout.read().decode().strip()
                    self.ssh_remote_path_var.set(actual_path)
                else:
                    error = stderr.read().decode().strip()
                    if not silent:
                        messagebox.showwarning("パス警告", f"指定されたリモートパスにアクセスできません: {error}")
                    # ホームディレクトリを代わりに使用
                    stdin, stdout, stderr = self.ssh_client.exec_command("echo $HOME")
                    home_dir = stdout.read().decode().strip()
                    self.ssh_remote_path_var.set(home_dir)
            else:
                # リモートパスが指定されていない場合はホームディレクトリを使用
                stdin, stdout, stderr = self.ssh_client.exec_command("echo $HOME")
                home_dir = stdout.read().decode().strip()
                self.ssh_remote_path_var.set(home_dir)
            
            # 設定を保存（接続成功時）
            self.save_config()
            
            if not silent:
                messagebox.showinfo("接続成功", f"{hostname}に接続しました。")
            
        except Exception as e:
            self.ssh_status_var.set("未接続")
            if not silent:
                messagebox.showerror("接続エラー", f"SSHサーバーへの接続に失敗しました: {str(e)}")
            # エラーログを出力
            self.debug_log(f"SSH接続エラー: {str(e)}")
    
    def disconnect_ssh(self, silent=False):
        """SSHサーバーから切断する"""
        if self.ssh_client:
            try:
                self.ssh_client.close()
                if not silent:
                    messagebox.showinfo("切断", "SSHサーバーから切断しました。")
            except Exception as e:
                self.debug_log(f"SSH切断エラー: {str(e)}")
            finally:
                self.ssh_client = None
                self.is_ssh_connected = False
                self.ssh_status_var.set("未接続")
                # ボタンの状態を更新
                if hasattr(self, "connect_button"):
                    self.connect_button.config(state=tk.NORMAL)
                if hasattr(self, "disconnect_button"):
                    self.disconnect_button.config(state=tk.DISABLED)
                
                # リモート実行ラジオボタンの状態を更新
                self.update_remote_execution_state()
    
    def auto_reconnect(self):
        """保存されたSSH情報で確認なしに自動的に再接続する"""
        hostname = self.ssh_hostname_var.get()
        username = self.ssh_username_var.get()
        if hostname and username:
            self.debug_log(f"SSHに自動再接続します: {username}@{hostname}")
            self.connect_ssh(silent=True)
    
    def execute_remote_command(self, command):
        """リモートサーバーでコマンドを実行し、その出力を返す"""
        if not self.is_ssh_connected or not self.ssh_client:
            raise Exception("SSH接続がありません。")
            
        stdin, stdout, stderr = self.ssh_client.exec_command(command)
        # 標準出力と標準エラー出力を取得
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        
        # 終了コードを取得
        exit_status = stdout.channel.recv_exit_status()
        
        return {
            "stdout": out,
            "stderr": err,
            "exit_code": exit_status
        }

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

def main():
    root = tk.Tk()
    app = ProcessControlTerminal(root)
    root.mainloop()

if __name__ == "__main__":
    main()
