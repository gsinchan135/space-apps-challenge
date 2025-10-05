import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except:
    DND_AVAILABLE = False

APP_TITLE = "Exo Hunter"

def train_model_backend(df, target, features, hyperparams, callback):
    import time
    time.sleep(2)
    fake_results = {
        'accuracy': 0.91,
        'report': 'Example classification report...',
        'confusion_matrix': np.array([[32,5],[3,40]])
    }
    callback(fake_results)

class ExoHunterApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.df = None
        self.current_page = 0
        self.page_size = 50

        self.build_ui()

    def build_ui(self):
        self.root.geometry('1200x800')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # Data preview
        self.root.rowconfigure(2, weight=0)  # Pagination
        self.root.rowconfigure(3, weight=1)  # Bottom controls

        # --- Top: Drop CSV ---
        self.drop_label = ttk.Label(
            self.root,
            text="Drop CSV here or click to browse",
            relief='sunken',
            padding=10
        )
        self.drop_label.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        self.drop_label.bind('<Button-1>', lambda e: self.open_file())
        if DND_AVAILABLE:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.handle_drop)

        # --- Data preview ---
        preview_frame = ttk.Frame(self.root)
        preview_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(preview_frame, show='headings')
        yscroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.tree.yview)
        xscroll = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        yscroll.grid(row=0, column=1, sticky='ns')
        xscroll.grid(row=1, column=0, sticky='ew')

        # --- Pagination controls ---
        nav_frame = ttk.Frame(self.root)
        nav_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        self.prev_btn = ttk.Button(nav_frame, text='⟨ Prev', command=self.prev_page, state='disabled')
        self.prev_btn.pack(side='left')
        self.page_label = ttk.Label(nav_frame, text='Page 1 / 1')
        self.page_label.pack(side='left', expand=True)
        self.next_btn = ttk.Button(nav_frame, text='Next ⟩', command=self.next_page, state='disabled')
        self.next_btn.pack(side='right')

        # --- Bottom controls ---
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.grid(row=3, column=0, sticky='nsew', padx=10, pady=5)
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=1)

        # Left: Target & features
        left_controls = ttk.Frame(bottom_frame)
        left_controls.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        ttk.Label(left_controls, text='Target Column:').pack(anchor='w')
        self.target_combo = ttk.Combobox(left_controls, state='readonly')
        self.target_combo.pack(fill='x', pady=2)
        self.target_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_feature_list())
        ttk.Label(left_controls, text='Feature Columns:').pack(anchor='w', pady=(5,0))
        self.features_listbox = tk.Listbox(left_controls, selectmode='multiple', height=8)
        self.features_listbox.pack(fill='both', expand=True, pady=2)

        # Middle: Hyperparameters
        middle_controls = ttk.LabelFrame(bottom_frame, text='Hyperparameters')
        middle_controls.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        ttk.Label(middle_controls, text='n_estimators:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.n_estimators_spin = tk.Spinbox(middle_controls, from_=10, to=2000, increment=10)
        self.n_estimators_spin.delete(0,'end'); self.n_estimators_spin.insert(0,'100')
        self.n_estimators_spin.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(middle_controls, text='max_depth (0=None):').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.max_depth_spin = tk.Spinbox(middle_controls, from_=0, to=100, increment=1)
        self.max_depth_spin.delete(0,'end'); self.max_depth_spin.insert(0,'0')
        self.max_depth_spin.grid(row=1, column=1, sticky='ew', padx=5, pady=2)

        # Right: Accuracy + Train button + confusion matrix
        right_controls = ttk.Frame(bottom_frame)
        right_controls.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)
        self.acc_label = ttk.Label(right_controls, text='Accuracy: N/A')
        self.acc_label.pack(anchor='w', pady=2)
        self.train_btn = ttk.Button(right_controls, text='Train Model', command=self.trigger_backend_train)
        self.train_btn.pack(pady=5)

        # Confusion matrix below train
        self.fig = Figure(figsize=(4,3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_controls)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # --- File handling ---
    def handle_drop(self, event):
        path = event.data.strip('{}')
        self.open_file(path)

    def open_file(self, path=None):
        if not path:
            path = filedialog.askopenfilename(title='Select CSV', filetypes=[('CSV files','*.csv')])
        if not path: return
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load CSV: {e}')
            return
        self.current_page = 0
        self.refresh_preview()
        self.populate_target_combo()

    # --- Preview / Pagination ---
    def refresh_preview(self):
        for c in self.tree.get_children():
            self.tree.delete(c)
        if self.df is None: return

        total_rows = len(self.df)
        total_pages = max(1, (total_rows + self.page_size - 1) // self.page_size)
        start = self.current_page * self.page_size
        end = start + self.page_size
        subset = self.df.iloc[start:end]

        cols = list(self.df.columns[:20])
        self.tree['columns'] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='w')

        for _, row in subset.iterrows():
            self.tree.insert('', 'end', values=[str(row[c]) for c in cols])

        self.page_label.config(text=f"Page {self.current_page+1} / {total_pages}")
        self.prev_btn.config(state='normal' if self.current_page>0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_page<total_pages-1 else 'disabled')

    def next_page(self):
        if self.df is None: return
        total_pages = max(1, (len(self.df)+self.page_size-1)//self.page_size)
        if self.current_page < total_pages-1:
            self.current_page += 1
            self.refresh_preview()

    def prev_page(self):
        if self.df is None: return
        if self.current_page>0:
            self.current_page -= 1
            self.refresh_preview()

    # --- Target / Features ---
    def populate_target_combo(self):
        if self.df is None: return
        cols = list(self.df.columns)
        self.target_combo['values'] = cols
        self.target_combo.set(cols[-1])
        self.refresh_feature_list()

    def refresh_feature_list(self):
        if self.df is None: return
        target = self.target_combo.get()
        self.features_listbox.delete(0,'end')
        for c in self.df.columns:
            if c != target:
                self.features_listbox.insert('end', c)

    # --- Training ---
    def trigger_backend_train(self):
        if self.df is None:
            messagebox.showwarning('No data','Load CSV first')
            return
        target = self.target_combo.get()
        features = [self.features_listbox.get(i) for i in self.features_listbox.curselection()]
        if not target or not features:
            messagebox.showwarning('Incomplete','Select target and at least one feature')
            return
        hyperparams = {
            'n_estimators': int(self.n_estimators_spin.get()),
            'max_depth': int(self.max_depth_spin.get()) or None
        }
        threading.Thread(
            target=train_model_backend,
            args=(self.df,target,features,hyperparams,self.receive_results)
        ).start()

    def receive_results(self, results):
        self.acc_label.config(text=f"Accuracy: {results['accuracy']:.4f}")
        self.update_confusion_matrix(results['confusion_matrix'])

    def update_confusion_matrix(self, cm):
        self.ax.clear()
        self.ax.set_title('Confusion Matrix')
        self.ax.imshow(cm, cmap='Blues')
        for (i,j), val in np.ndenumerate(cm):
            self.ax.text(j,i,str(val),ha='center',va='center',color='black')
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = ExoHunterApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
