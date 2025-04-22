from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog, ttk
from brian2 import *
import tkinter as tk
import os
import json
from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import numpy as np
import sys
import traceback
from tkinter.scrolledtext import ScrolledText
from tqdm import tqdm
import io
import time
import threading
import queue

class ValidatorGUI:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Validator")
        self.root.geometry("800x550")

        self.network_dir = None
        self.params_path = None
        self.params = None

        # ドロップフレーム
        self.dir_drop_frame = tk.Frame(self.root, bg="lightgreen", relief="groove", borderwidth=2)
        self.dir_drop_frame.drop_target_register(DND_FILES)
        self.dir_drop_frame.dnd_bind('<<Drop>>', self.handle_drop)
        self.dir_drop_frame.bind('<Button-1>', lambda e: self.open_file())
        self.dir_drop_frame.pack(fill="both", expand=True)
        self.lb1 = tk.Label(self.dir_drop_frame, text="Drop network directory or params json file here!", bg="lightgreen", font=("Arial", 17, "bold"))
        self.lb1.place(relx=0.5, rely=0.5, anchor="center")

        # 設定レーム
        self.setting_frame = tk.Frame(self.root, bg="lightgray", relief="groove", borderwidth=2)
        self.setting_frame.pack(fill="both", expand=True)
        
        # バリデーション名入力
        self.lb3 = tk.Label(self.setting_frame, text="Validation name:", bg="lightgray", font=("Arial", 10))
        self.validation_name = tk.Entry(self.setting_frame, width=50)
        self.lb3.place(relx=0.05, rely=0.1, anchor="w")
        self.validation_name.place(relx=0.3, rely=0.1, anchor="w")
        
        # ロードされたディレクトリ表示
        self.lb4 = tk.Label(self.setting_frame, text="loaded directory:", bg="lightgray", font=("Arial", 10))
        self.loaded_dir = tk.Label(self.setting_frame, text="None", bg="lightgray", font=("Arial", 10))
        self.lb4.place(relx=0.05, rely=0.3, anchor="w")
        self.loaded_dir.place(relx=0.3, rely=0.3, anchor="w")
        
        # ロードされたパラメータ表示
        self.lb5 = tk.Label(self.setting_frame, text="loaded params:", bg="lightgray", font=("Arial", 10))
        self.loaded_params = tk.Label(self.setting_frame, text="None", bg="lightgray", font=("Arial", 10))
        self.lb5.place(relx=0.05, rely=0.5, anchor="w")
        self.loaded_params.place(relx=0.3, rely=0.5, anchor="w")

        # 実行ボタン
        self.runbutton_frame = tk.Frame(self.root, bg="lightgray", relief="groove", borderwidth=2)
        self.runbutton_frame.pack(fill="both")
        self.bt3 = tk.Button(self.runbutton_frame, text="Run Validation!", font=("Arial", 15, "bold"), command=self.validate, height=2)
        self.bt3.pack(fill="both", expand=True)

        # プログレスバレームを削除し、代わりにログ表示用のテキストエリアを追加
        self.log_frame = tk.Frame(self.root, bg="lightgray", relief="groove", borderwidth=2)
        self.log_frame.pack(fill="both", expand=True)
        
        self.log_text = ScrolledText(self.log_frame, height=10, bg="black", fg="white")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 標準出力をリダイレクト
        sys.stdout = self.TextRedirector(self.log_text)
        sys.stderr = self.TextRedirector(self.log_text, "error")

        # メッセージキューを追加
        self.message_queue = queue.Queue()
        self.root.after(100, self.process_queue)

    def process_queue(self):
        """メッセージキューを処理"""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if isinstance(msg, Exception):
                    tk.messagebox.showerror("Error", str(msg))
                    self.bt3.config(state="normal")
                elif msg == "COMPLETE":
                    tk.messagebox.showinfo("Success", "Validation completed!\n\nvalidation name: " + self.validation_name.get())
                    self.bt3.config(state="normal")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def run_validation(self):
        """別スレッドで実行される検証処理"""
        try:
            np.random.seed(self.params["seed"])
            
            validator = Validator(
                target_path=self.network_dir,
                assigned_labels_path=f"{self.network_dir}/assigned_labels.pkl",
                params=self.params,
                network_type=self.params["network_type"],
                enable_monitor=self.params["enable_monitor"]
            )
            
            validator.validate(
                n_samples=self.params["n_samples"], 
                examination_name=self.validation_name.get()
            )
            
            self.message_queue.put("COMPLETE")
            
        except Exception as e:
            self.message_queue.put(e)

    def validate(self):
        if not self.network_dir or not self.params_path:
            tk.messagebox.showerror("Error", "Please load both network directory and params file.")
            return
            
        if not self.validation_name.get():
            tk.messagebox.showerror("Error", "Please enter validation name.")
            return
        
        # 実行中はボタンを無効化
        self.bt3.config(state="disabled")
        
        # 別スレッドで検証を実行
        thread = threading.Thread(target=self.run_validation)
        thread.daemon = True  # メインスレッド終了時に一緒に終了
        thread.start()

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Open network directory or params json file",
            filetypes=[("Network directory", ""), ("Params json file", "*.json")]
        )
        self.handle_path(file_path)

    def handle_drop(self, event):
        self.handle_path(event.data)

    def handle_path(self, path):
        path = path.strip('{}')  # TkinterDnDの仕様で{}が付くことがあるので除去
        
        if path.endswith('.json'):
            self.params_path = path
            self.params = tools.load_parameters(path)
            self.loaded_params.config(text=os.path.basename(path))
            self.show_json_content()
        else:
            self.network_dir = path
            self.loaded_dir.config(text=os.path.basename(path))

    def show_json_content(self):
        # 新しいウィンドウを作成
        json_window = tk.Toplevel(self.root)
        json_window.title("JSON Parameters")
        json_window.geometry("600x400")

        # テキストウィジェットを作成してJSONの内容を表示
        text_widget = tk.Text(json_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # スクロールバーを追加
        scrollbar = tk.Scrollbar(json_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # スクロールバーとテキストウィジェットを連動
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

        def convert_quantities(obj):
            if hasattr(obj, 'dimensions'):  # Quantityオブジェクトの場合
                return str(obj)
            elif isinstance(obj, dict):  # 辞書の場合
                return {key: convert_quantities(value) for key, value in obj.items()}
            elif isinstance(obj, list):  # リストの場合
                return [convert_quantities(item) for item in obj]
            return obj

        # パラメータを再帰的に変換
        params_display = convert_quantities(self.params)

        # JSONの内容を整形して表示
        formatted_json = json.dumps(params_display, indent=4, ensure_ascii=False)
        text_widget.insert(tk.END, formatted_json)
        text_widget.config(state=tk.DISABLED)  # 読み取り専用に設定

    class TextRedirector:
        def __init__(self, widget, tag="stdout"):
            self.widget = widget
            self.tag = tag
            self.buffer = io.StringIO()
            self.last_update = time.time()
            self.queue = queue.Queue()
            widget.after(100, self.process_queue)

        def process_queue(self):
            """GUIスレッドでテキスト更新を処理"""
            try:
                while True:
                    text = self.queue.get_nowait()
                    self.widget.configure(state="normal")
                    
                    if '\r' in text:
                        lines = text.split('\r')
                        text = lines[-1]
                        
                        last_line_start = self.widget.index("end-2c linestart")
                        last_line_end = self.widget.index("end-1c")
                        if "%" in self.widget.get(last_line_start, last_line_end):
                            self.widget.delete(last_line_start, last_line_end)
                    
                    self.widget.insert("end", text, (self.tag,))
                    self.widget.see("end")
                    self.widget.configure(state="disabled")
            except queue.Empty:
                pass
            finally:
                self.widget.after(100, self.process_queue)

        def write(self, str):
            self.queue.put(str)

        def flush(self):
            pass

def main():
    # ターミナルを非表示にする
    if sys.platform.startswith('win'):
        import win32gui
        import win32con
        # 現在のプロセスのウィンドウハンドルを取得
        hwnd = win32gui.GetForegroundWindow()
        # ウィンドウを非表示に
        win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
        
        # コマンドプロンプトも非表示に（親プロセスがcmdの場合）
        try:
            parent = win32gui.GetParent(hwnd)
            if parent:
                win32gui.ShowWindow(parent, win32con.SW_HIDE)
        except:
            pass
    
    app = ValidatorGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()
