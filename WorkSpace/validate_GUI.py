from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
from brian2 import *
import tkinter as tk
import os
import json
from Brian2_Framework.Validator import Validator
import Brian2_Framework.Tools as tools
import numpy as np

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

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Open network directory or params json file",
            filetypes=[("Network directory", ""), ("Params json file", "*.json")]
        )
        self.handle_path(file_path)

    def validate(self):
        if not self.network_dir or not self.params_path:
            tk.messagebox.showerror("Error", "Please load both network directory and params file.")
            return
            
        if not self.validation_name.get():
            tk.messagebox.showerror("Error", "Please enter validation name.")
            return
            
        np.random.seed(self.params["seed"])
        
        self.root.withdraw() # メインウィンドウを非表示にする
        # JSONパラメータ表示ウィンドウを閉じる
        for window in self.root.winfo_children():
            if isinstance(window, tk.Toplevel):
                window.destroy()
        
        validator = Validator(
            target_path=self.network_dir,
            assigned_labels_path=f"{self.network_dir}/assigned_labels.pkl",
            params=self.params,
            network_type=self.params["network_type"],
            enable_monitor=self.params["enable_monitor"]
        )
        
        validator.validate(n_samples=self.params["n_samples"], examination_name=self.validation_name.get())
        tk.messagebox.showinfo("Success", "Validation completed!\n\nvalidation name: " + self.validation_name.get())

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

def main():
    app = ValidatorGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()
