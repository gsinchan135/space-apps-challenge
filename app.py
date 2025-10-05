import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Set matplotlib backend for tkinter
plt.style.use('default')

# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except:
    DND_AVAILABLE = False

# Import our ML model
try:
    from final_model import FinalExoplanetModel
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML model not available: {e}")

APP_TITLE = "Exo Hunter - AI Exoplanet Discovery"

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
        
        # Initialize ML model
        self.ml_model = None
        self.ml_predictions = None
        self.ml_results_ready = False
        
        # Initialize ML model in background
        self.init_ml_model()
        
        self.build_ui()

    def init_ml_model(self):
        """Initialize the ML model in background"""
        def load_model():
            try:
                self.ml_model = FinalExoplanetModel(output_dir="app_ml_outputs")
                # Try to load pre-trained model
                try:
                    self.ml_model.load_model('final_exoplanet_model.joblib')
                    print("✅ Pre-trained model loaded successfully!")
                except:
                    print("⚠️ No pre-trained model found. Will train when data is loaded.")
            except Exception as e:
                print(f"❌ Error initializing ML model: {e}")
        
        # Load in background thread
        threading.Thread(target=load_model, daemon=True).start()

    def build_ui(self):
        self.root.geometry('1400x900')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.ml_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="📊 Data Explorer")
        self.notebook.add(self.ml_tab, text="🤖 AI Predictions")
        
        # Build data tab
        self.build_data_tab()
        
        # Build ML tab
        self.build_ml_tab()

    def build_data_tab(self):
        """Build the original data exploration tab"""
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.rowconfigure(1, weight=1)  # Data preview
        self.data_tab.rowconfigure(2, weight=0)  # Pagination
        self.data_tab.rowconfigure(3, weight=1)  # Bottom controls

        # --- Top: Drop CSV ---
        self.drop_label = ttk.Label(
            self.data_tab,
            text="🚀 Drop exoplanet CSV here or click to browse",
            relief='sunken',
            padding=15,
            font=('Arial', 12)
        )
        self.drop_label.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        self.drop_label.bind('<Button-1>', lambda e: self.open_file())
        if DND_AVAILABLE:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.handle_drop)

        # --- Data preview ---
        preview_frame = ttk.LabelFrame(self.data_tab, text="Data Preview")
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
        nav_frame = ttk.Frame(self.data_tab)
        nav_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        self.prev_btn = ttk.Button(nav_frame, text='⟨ Previous', command=self.prev_page, state='disabled')
        self.prev_btn.pack(side='left')
        self.page_label = ttk.Label(nav_frame, text='Page 1 / 1', font=('Arial', 10))
        self.page_label.pack(side='left', expand=True)
        self.next_btn = ttk.Button(nav_frame, text='Next ⟩', command=self.next_page, state='disabled')
        self.next_btn.pack(side='right')

        # --- Bottom controls ---
        bottom_frame = ttk.Frame(self.data_tab)
        bottom_frame.grid(row=3, column=0, sticky='nsew', padx=10, pady=5)
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=1)

        # Left: Target & features
        left_controls = ttk.LabelFrame(bottom_frame, text="Model Configuration")
        left_controls.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        ttk.Label(left_controls, text='Target Column:').pack(anchor='w')
        self.target_combo = ttk.Combobox(left_controls, state='readonly')
        self.target_combo.pack(fill='x', pady=2)
        self.target_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_feature_list())
        ttk.Label(left_controls, text='Feature Columns:').pack(anchor='w', pady=(5,0))
        self.features_listbox = tk.Listbox(left_controls, selectmode='multiple', height=8)
        self.features_listbox.pack(fill='both', expand=True, pady=2)

        # Middle: Hyperparameters
        middle_controls = ttk.LabelFrame(bottom_frame, text='AI Model Settings')
        middle_controls.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        ttk.Label(middle_controls, text='n_estimators:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.n_estimators_spin = tk.Spinbox(middle_controls, from_=10, to=2000, increment=10)
        self.n_estimators_spin.delete(0,'end'); self.n_estimators_spin.insert(0,'200')
        self.n_estimators_spin.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(middle_controls, text='max_depth (0=None):').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.max_depth_spin = tk.Spinbox(middle_controls, from_=0, to=100, increment=1)
        self.max_depth_spin.delete(0,'end'); self.max_depth_spin.insert(0,'6')
        self.max_depth_spin.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(middle_controls, text='learning_rate:').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.learning_rate_entry = tk.Entry(middle_controls)
        self.learning_rate_entry.insert(0, '0.1')
        self.learning_rate_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=2)

        # Right: Training controls
        right_controls = ttk.LabelFrame(bottom_frame, text="AI Training")
        right_controls.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)
        self.acc_label = ttk.Label(right_controls, text='🎯 Accuracy: Ready to train', font=('Arial', 10, 'bold'))
        self.acc_label.pack(anchor='w', pady=2)
        self.train_btn = ttk.Button(right_controls, text='🚀 Train AI Model', command=self.trigger_ml_training)
        self.train_btn.pack(pady=5)
        self.predict_btn = ttk.Button(right_controls, text='🔮 Make Predictions', command=self.trigger_ml_predictions, state='disabled')
        self.predict_btn.pack(pady=2)

        # Progress bar
        self.progress = ttk.Progressbar(right_controls, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)

    def build_ml_tab(self):
        """Build the ML results tab with the requested layout"""
        self.ml_tab.columnconfigure(0, weight=1)
        self.ml_tab.columnconfigure(1, weight=1)
        self.ml_tab.rowconfigure(0, weight=1)
        self.ml_tab.rowconfigure(1, weight=1)

        # Top left: Prediction frequency plot
        freq_frame = ttk.LabelFrame(self.ml_tab, text="🔥 Prediction Class Distribution")
        freq_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        self.freq_fig = Figure(figsize=(8, 5), dpi=80)
        self.freq_ax = self.freq_fig.add_subplot(111)
        self.freq_canvas = FigureCanvasTkAgg(self.freq_fig, master=freq_frame)
        self.freq_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # Top right: Predictions table
        table_frame = ttk.LabelFrame(self.ml_tab, text="🌟 Exoplanet Predictions")
        table_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # Create treeview for predictions
        self.pred_tree = ttk.Treeview(table_frame, columns=('Planet', 'Class', 'Confidence'), show='headings')
        self.pred_tree.heading('Planet', text='🪐 Planet Name')
        self.pred_tree.heading('Class', text='🎯 Predicted Class')
        self.pred_tree.heading('Confidence', text='📊 Confidence %')
        
        self.pred_tree.column('Planet', width=150)
        self.pred_tree.column('Class', width=120)
        self.pred_tree.column('Confidence', width=100)

        pred_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.pred_tree.yview)
        self.pred_tree.configure(yscrollcommand=pred_scroll.set)
        
        self.pred_tree.grid(row=0, column=0, sticky='nsew')
        pred_scroll.grid(row=0, column=1, sticky='ns')

        # Bottom: Feature importance plot (spans both columns)
        importance_frame = ttk.LabelFrame(self.ml_tab, text="⭐ Most Important Features for Exoplanet Detection")
        importance_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        
        self.importance_fig = Figure(figsize=(12, 5), dpi=80)
        self.importance_ax = self.importance_fig.add_subplot(111)
        self.importance_canvas = FigureCanvasTkAgg(self.importance_fig, master=importance_frame)
        self.importance_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # ML Status label
        self.ml_status_label = ttk.Label(self.ml_tab, text="🤖 Ready for AI predictions - Load data and train model first!", 
                                        font=('Arial', 11, 'bold'), foreground='blue')
        self.ml_status_label.grid(row=2, column=0, columnspan=2, pady=5)

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
    def trigger_ml_training(self):
        """Train the actual ML model using our FinalExoplanetModel"""
        if self.df is None:
            messagebox.showwarning('No data','Load CSV first')
            return
        
        if self.ml_model is None:
            messagebox.showerror('ML Error', 'ML model not initialized. Please restart the app.')
            return
        
        # Start progress bar
        self.progress.start()
        self.train_btn.config(state='disabled', text='🔄 Training...')
        self.acc_label.config(text='🔄 Training AI model...')
        
        def train_thread():
            try:
                # Save current dataframe temporarily for training
                temp_file = 'temp_training_data.csv'
                self.df.to_csv(temp_file, index=False)
                
                # Get hyperparameters from UI
                hyperparams = {
                    'n_estimators': int(self.n_estimators_spin.get()),
                    'max_depth': int(self.max_depth_spin.get()) or None,
                    'learning_rate': float(self.learning_rate_entry.get()),
                    'random_state': 42
                }
                
                # Update model hyperparameters
                self.ml_model.hyperparameters.update(hyperparams)
                
                # Load and train
                X, y, _ = self.ml_model.load_and_prepare_data(temp_file)
                results = self.ml_model.train_model(X, y)
                
                # Clean up temp file
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.training_complete(results))
                
            except Exception as e:
                self.root.after(0, lambda: self.training_error(str(e)))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def training_complete(self, results):
        """Handle training completion"""
        self.progress.stop()
        self.train_btn.config(state='normal', text='🚀 Train AI Model')
        self.predict_btn.config(state='normal')
        
        accuracy = results.get('validation_accuracy', 0)
        f1_score = results.get('validation_f1', 0)
        
        self.acc_label.config(text=f'🎯 Accuracy: {accuracy:.3f} | F1: {f1_score:.3f}')
        self.ml_status_label.config(text='✅ AI model trained successfully! Switch to AI Predictions tab to see results.', 
                                   foreground='green')
        
        messagebox.showinfo('Training Complete', 
                           f'🎉 AI model trained successfully!\n\n'
                           f'📊 Accuracy: {accuracy:.3f}\n'
                           f'📈 F1-Score: {f1_score:.3f}\n\n'
                           f'Switch to "AI Predictions" tab to see results!')
    
    def training_error(self, error_msg):
        """Handle training error"""
        self.progress.stop()
        self.train_btn.config(state='normal', text='🚀 Train AI Model')
        self.acc_label.config(text='❌ Training failed')
        messagebox.showerror('Training Error', f'Failed to train model:\n{error_msg}')
    
    def trigger_ml_predictions(self):
        """Make predictions using the trained model"""
        if self.ml_model is None or not self.ml_model.is_trained:
            messagebox.showwarning('Model not ready', 'Please train the model first')
            return
        
        if self.df is None:
            messagebox.showwarning('No data', 'Load CSV data first')
            return
        
        self.predict_btn.config(state='disabled', text='🔄 Predicting...')
        self.ml_status_label.config(text='🔄 AI is analyzing exoplanet data...', foreground='orange')
        
        def predict_thread():
            try:
                # Save data temporarily
                temp_file = 'temp_prediction_data.csv'
                self.df.to_csv(temp_file, index=False)
                
                # Get UI-friendly predictions
                ui_data = self.ml_model.predict_with_simple_format(temp_file, 'file')
                
                # Clean up
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                # Update UI
                self.root.after(0, lambda: self.predictions_complete(ui_data))
                
            except Exception as e:
                self.root.after(0, lambda: self.predictions_error(str(e)))
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def predictions_complete(self, ui_data):
        """Handle prediction completion and update ML tab"""
        self.predict_btn.config(state='normal', text='🔮 Make Predictions')
        self.ml_predictions = ui_data
        self.ml_results_ready = True
        
        # Update ML tab with results
        self.update_predictions_table(ui_data)
        self.update_frequency_plot(ui_data)
        self.update_feature_importance()
        
        total = ui_data['total_predictions']
        high_conf = ui_data['high_confidence_count']
        
        self.ml_status_label.config(
            text=f'🎉 Found {total} objects! {high_conf} high-confidence predictions. Check the plots below!', 
            foreground='green'
        )
        
        # Switch to ML tab
        self.notebook.select(self.ml_tab)
        
        messagebox.showinfo('Predictions Complete', 
                           f'🌟 AI analysis complete!\n\n'
                           f'📊 Total objects analyzed: {total}\n'
                           f'🎯 High confidence predictions: {high_conf}\n\n'
                           f'Check the "AI Predictions" tab for detailed results!')
    
    def predictions_error(self, error_msg):
        """Handle prediction error"""
        self.predict_btn.config(state='normal', text='🔮 Make Predictions')
        self.ml_status_label.config(text='❌ Prediction failed', foreground='red')
        messagebox.showerror('Prediction Error', f'Failed to make predictions:\n{error_msg}')
    
    def update_predictions_table(self, ui_data):
        """Update the predictions table in ML tab"""
        # Clear existing data
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)
        
        # Add predictions (limit to first 100 for performance)
        predictions = ui_data['predictions'][:100]
        
        for pred in predictions:
            planet_name = pred['exoplanet_name']
            pred_class = pred['predicted_class']
            confidence = f"{pred['confidence']:.1%}"
            
            # Color code based on class
            if pred_class == 'CONFIRMED':
                tags = ('confirmed',)
            elif pred_class == 'FALSE POSITIVE':
                tags = ('false_positive',)
            else:
                tags = ('candidate',)
            
            self.pred_tree.insert('', 'end', values=(planet_name, pred_class, confidence), tags=tags)
        
        # Configure row colors
        self.pred_tree.tag_configure('confirmed', background='lightgreen')
        self.pred_tree.tag_configure('false_positive', background='lightcoral')
        self.pred_tree.tag_configure('candidate', background='lightyellow')
        
        if len(predictions) >= 100:
            self.pred_tree.insert('', 'end', values=('...', f'and {len(ui_data["predictions"])-100} more', '...'))
    
    def update_frequency_plot(self, ui_data):
        """Update the frequency plot in ML tab"""
        self.freq_ax.clear()
        
        # Get class summary
        class_summary = ui_data['class_summary']
        classes = list(class_summary.keys())
        counts = [class_summary[cls]['count'] for cls in classes]
        percentages = [class_summary[cls]['percentage'] for cls in classes]
        
        # Color mapping with better colors
        colors = {'CONFIRMED': '#2E8B57', 'FALSE POSITIVE': '#DC143C', 'CANDIDATE': '#FFD700'}
        bar_colors = [colors.get(cls, '#4169E1') for cls in classes]
        
        # Create bar plot with better spacing
        bars = self.freq_ax.bar(classes, counts, color=bar_colors, alpha=0.8, 
                               edgecolor='black', linewidth=1.2, width=0.6)
        
        # Add count and percentage labels with better positioning
        max_count = max(counts) if counts else 1
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            height = bar.get_height()
            # Position labels higher above bars
            self.freq_ax.text(bar.get_x() + bar.get_width()/2., height + max_count*0.03,
                            f'{count}\n({pct:.1f}%)', 
                            ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Improve styling
        self.freq_ax.set_title('🪐 Exoplanet Classification Results', fontsize=14, fontweight='bold', pad=20)
        self.freq_ax.set_ylabel('Number of Objects', fontsize=11, fontweight='bold')
        self.freq_ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
        
        # Add grid for better readability
        self.freq_ax.grid(axis='y', alpha=0.3, linestyle='--')
        self.freq_ax.set_axisbelow(True)
        
        # Set y-axis limit to accommodate labels
        if counts:
            self.freq_ax.set_ylim(0, max_count * 1.15)
        
        # Improve x-axis labels
        plt.setp(self.freq_ax.get_xticklabels(), rotation=0, ha='center', fontweight='bold')
        
        # Better layout with more padding
        self.freq_fig.tight_layout(pad=2.0)
        self.freq_canvas.draw()
    
    def update_feature_importance(self):
        """Update the feature importance plot"""
        if not self.ml_model or not self.ml_model.is_trained:
            return
        
        self.importance_ax.clear()
        
        try:
            # Get feature importance from model
            importance_df = self.ml_model.get_feature_importance(top_n=15)
            
            # Create horizontal bar plot with better styling
            y_pos = np.arange(len(importance_df))
            bars = self.importance_ax.barh(y_pos, importance_df['importance'], 
                                         color='lightcoral', alpha=0.8, 
                                         edgecolor='darkred', linewidth=1)
            
            # Customize plot with better styling
            self.importance_ax.set_yticks(y_pos)
            self.importance_ax.set_yticklabels(importance_df['feature'], fontsize=10, fontweight='bold')
            self.importance_ax.set_xlabel('Feature Importance Score', fontsize=11, fontweight='bold')
            self.importance_ax.set_title('🔍 Top Features for Exoplanet Detection', fontsize=14, fontweight='bold', pad=15)
            
            # Add grid for better readability
            self.importance_ax.grid(axis='x', alpha=0.3, linestyle='--')
            self.importance_ax.set_axisbelow(True)
            
            # Add value labels with better formatting
            max_importance = max(importance_df['importance']) if len(importance_df) > 0 else 1
            for i, bar in enumerate(bars):
                width = bar.get_width()
                self.importance_ax.text(width + max_importance*0.01, 
                                      bar.get_y() + bar.get_height()/2,
                                      f'{width:.3f}', ha='left', va='center', 
                                      fontsize=9, fontweight='bold')
            
            self.importance_ax.invert_yaxis()  # Highest importance at top
            
            # Better layout with padding
            self.importance_fig.tight_layout(pad=2.0)
            self.importance_canvas.draw()
            
        except Exception as e:
            self.importance_ax.text(0.5, 0.5, f'⚠️ Feature importance\nnot available\n({str(e)})', 
                                  ha='center', va='center', transform=self.importance_ax.transAxes,
                                  fontsize=12, fontweight='bold')
            self.importance_canvas.draw()

    def trigger_backend_train(self):
        """Legacy method - redirects to new ML training"""
        self.trigger_ml_training()

    # --- File handling ---
    def handle_drop(self, event):
        path = event.data.strip('{}')
        self.open_file(path)

    def open_file(self, path=None):
        if not path:
            path = filedialog.askopenfilename(title='Select Exoplanet CSV', filetypes=[('CSV files','*.csv')])
        if not path: return
        try:
            self.df = pd.read_csv(path, comment='#')  # Handle NASA format with comments
            self.ml_status_label.config(text=f'📁 Loaded {len(self.df)} exoplanet candidates from {path.split("/")[-1]}', 
                                       foreground='blue')
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

        cols = list(self.df.columns[:20])  # Show first 20 columns
        self.tree['columns'] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='w')

        for _, row in subset.iterrows():
            self.tree.insert('', 'end', values=[str(row[c]) for c in cols])

        self.page_label.config(text=f"Page {self.current_page+1} / {total_pages} ({total_rows} total objects)")
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
        # Auto-select koi_disposition if available
        if 'koi_disposition' in cols:
            self.target_combo.set('koi_disposition')
        else:
            self.target_combo.set(cols[-1])
        self.refresh_feature_list()

    def refresh_feature_list(self):
        if self.df is None: return
        target = self.target_combo.get()
        self.features_listbox.delete(0,'end')
        for c in self.df.columns:
            if c != target:
                self.features_listbox.insert('end', c)

def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = ExoHunterApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
