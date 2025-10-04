import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

APP_TITLE = "exo hunter"

# This backend function will be implemented elsewhere and imported later
def train_model_backend(dataframe, target, features, hyperparams, callback):
    """Placeholder stub for backend call.
    The backend module should handle data processing and model training.
    Once done, it should call `callback(results_dict)` with the results.
    """
    import time
    time.sleep(2)  # Simulate training delay
    fake_results = {
        'accuracy': 0.91,
        'report': 'Example classification report...',
        'confusion_matrix': np.array([[32, 5], [3, 40]])
    }
    callback(fake_results)


class ExoHunterApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.df = None

        self.build_ui()

    def build_ui(self):
        self.root.geometry('1000x700')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # --- File Section ---
        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky='ew')

        self.drop_frame = ttk.LabelFrame(top, text="Load CSV")
        self.drop_frame.pack(fill='x', expand=True)
        self.drop_label = ttk.Label(self.drop_frame, text="Drop CSV here or click to browse", relief='sunken', anchor='center', padding=10)
        self.drop_label.pack(fill='x', padx=8, pady=8)
        self.drop_label.bind('<Button-1>', lambda e: self.open_file())

        if DND_AVAILABLE:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.handle_drop)

        # --- Data Preview ---
        mid = ttk.Frame(self.root, padding=10)
        mid.grid(row=1, column=0, sticky='nsew')
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(0, weight=1)

        # CSV Preview
        left = ttk.Frame(mid)
        left.grid(row=0, column=0, sticky='nsew', padx=(0,10))

        ttk.Label(left, text='Data Preview (first 50 rows):').pack(anchor='w')
        self.tree = ttk.Treeview(left, show='headings', height=15)
        self.tree.pack(fill='both', expand=True)

        # --- Controls ---
        right = ttk.Frame(mid)
        right.grid(row=0, column=1, sticky='ns')

        ttk.Label(right, text='Target Column:').pack(anchor='w')
        self.target_combo = ttk.Combobox(right, state='readonly')
        self.target_combo.pack(fill='x')
        self.target_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_feature_list())

        ttk.Label(right, text='Feature Columns:').pack(anchor='w', pady=(8,0))
        self.features_listbox = tk.Listbox(right, selectmode='multiple', height=8)
        self.features_listbox.pack(fill='both', expand=True)

        # Hyperparameter Frame
        hp_frame = ttk.LabelFrame(right, text='Hyperparameters', padding=6)
        hp_frame.pack(fill='x', pady=(8,4))

        ttk.Label(hp_frame, text='n_estimators:').grid(row=0, column=0, sticky='w')
        self.n_estimators_spin = tk.Spinbox(hp_frame, from_=10, to=2000, increment=10)
        self.n_estimators_spin.delete(0, 'end'); self.n_estimators_spin.insert(0, '100')
        self.n_estimators_spin.grid(row=0, column=1, sticky='ew')

        ttk.Label(hp_frame, text='max_depth (0=None):').grid(row=1, column=0, sticky='w')
        self.max_depth_spin = tk.Spinbox(hp_frame, from_=0, to=100, increment=1)
        self.max_depth_spin.delete(0, 'end'); self.max_depth_spin.insert(0, '0')
        self.max_depth_spin.grid(row=1, column=1, sticky='ew')

        # Buttons
        btn_train = ttk.Button(right, text='Train Model', command=self.trigger_backend_train)
        btn_train.pack(fill='x', pady=(8,0))

        # --- Output / Metrics ---
        metrics = ttk.LabelFrame(right, text='Model Statistics', padding=6)
        metrics.pack(fill='both', expand=True, pady=(8,0))
        self.acc_label = ttk.Label(metrics, text='Accuracy: N/A')
        self.acc_label.pack(anchor='w')
        self.report_text = tk.Text(metrics, height=8, wrap='word', state='disabled')
        self.report_text.pack(fill='both', expand=True)

        # --- Confusion Matrix ---
        bottom = ttk.Frame(self.root, padding=10)
        bottom.grid(row=2, column=0, sticky='nsew')
        self.fig = Figure(figsize=(4,3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # --- Handlers ---

    def handle_drop(self, event):
        data = event.data
        if data.startswith('{') and data.endswith('}'):
            data = data[1:-1]
        path = data.split()[0]
        self.open_file(path)

    def open_file(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title='Select CSV', filetypes=[('CSV files','*.csv')])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load CSV: {e}')
            return
        self.refresh_preview()
        self.populate_target_combo()

    def refresh_preview(self):
        for c in self.tree.get_children():
            self.tree.delete(c)
        if self.df is None:
            return
        cols = list(self.df.columns[:20])
        self.tree['columns'] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='w')
        for i, row in self.df.head(50).iterrows():
            self.tree.insert('', 'end', values=[str(row[c]) for c in cols])

    def populate_target_combo(self):
        if self.df is None:
            return
        cols = list(self.df.columns)
        self.target_combo['values'] = cols
        self.target_combo.set(cols[-1])
        self.refresh_feature_list()

    def refresh_feature_list(self):
        if self.df is None:
            return
        target = self.target_combo.get()
        self.features_listbox.delete(0, 'end')
        for c in self.df.columns:
            if c != target:
                self.features_listbox.insert('end', c)

    def trigger_backend_train(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Please load a CSV first')
            return
        target = self.target_combo.get()
        features = [self.features_listbox.get(i) for i in self.features_listbox.curselection()]
        if not target or not features:
            messagebox.showwarning('Incomplete', 'Select target and at least one feature')
            return
        hyperparams = {
            'n_estimators': int(self.n_estimators_spin.get()),
            'max_depth': int(self.max_depth_spin.get()) or None
        }
        threading.Thread(target=train_model_backend, args=(self.df, target, features, hyperparams, self.receive_backend_results)).start()

    def receive_backend_results(self, results):
        self.acc_label.config(text=f"Accuracy: {results['accuracy']:.4f}")
        self.report_text.configure(state='normal')
        self.report_text.delete('1.0', 'end')
        self.report_text.insert('end', results['report'])
        self.report_text.configure(state='disabled')
        self.update_confusion_matrix(results['confusion_matrix'])

    def update_confusion_matrix(self, cm):
        self.ax.clear()
        self.ax.set_title('Confusion Matrix')
        self.ax.imshow(cm, cmap='Blues')
        for (i, j), val in np.ndenumerate(cm):
            self.ax.text(j, i, str(val), ha='center', va='center', color='black')
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    ExoHunterApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()