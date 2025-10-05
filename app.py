import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pygame
import os
import time

# Set matplotlib backend for tkinter
plt.style.use('default')

# Initialize pygame mixer for audio
pygame.mixer.init()

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

    def play_background_music(self):
        """Play the space.mp3 background music"""
        try:
            music_path = os.path.join(os.path.dirname(__file__), "space.mp3")
            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.3)  # Set volume to 30%
                pygame.mixer.music.play(-1)  # Play indefinitely
                print("🎵 Background music started")
            else:
                print("⚠️ space.mp3 not found")
        except Exception as e:
            print(f"⚠️ Background music could not be loaded: {e}")
            print("🎵 Continuing without background music...")

    def build_home_tab(self):
        """Build the home page with description and integrated visualizations"""
        # Home tab background is already set in build_ui as tk.Frame with bg
        
        # Main container with scrollbar
        canvas = tk.Canvas(self.home_tab, bg='#0a0f1f', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.home_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title section
        title_frame = tk.Frame(scrollable_frame, bg='#0a0f1f', height=150)
        title_frame.pack(fill='x', padx=20, pady=20)
        
        title_label = tk.Label(title_frame, 
                              text="🌌 ExoHunter - AI Exoplanet Discovery Tool 🚀",
                              font=('Arial', 28, 'bold'),
                              fg='#ffcc66',
                              bg='#0a0f1f')
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(title_frame,
                                 text="Hunting for Distant Worlds with Artificial Intelligence",
                                 font=('Arial', 16, 'italic'),
                                 fg='#66ccff',
                                 bg='#0a0f1f')
        subtitle_label.pack()
        
        # Problem description section
        desc_frame = tk.Frame(scrollable_frame, bg='#1a1f2f', relief='raised', bd=2)
        desc_frame.pack(fill='x', padx=20, pady=20)
        
        desc_title = tk.Label(desc_frame,
                             text="🎯 The Challenge: Finding Exoplanets",
                             font=('Arial', 20, 'bold'),
                             fg='#ffcc66',
                             bg='#1a1f2f')
        desc_title.pack(pady=15)
        
        desc_text = tk.Text(desc_frame, 
                           height=8, 
                           font=('Arial', 12),
                           fg='white',
                           bg='#1a1f2f',
                           wrap='word',
                           relief='flat',
                           state='normal')
        desc_text.pack(fill='x', padx=20, pady=10)
        
        problem_description = """🌟 Exoplanets are planets that orbit stars outside our solar system. With over 5,000 confirmed exoplanets discovered, we're constantly searching for more!

🔍 The Challenge: Traditional methods of finding exoplanets require manual analysis of massive datasets from space telescopes like Kepler. This process is time-consuming and can miss subtle planetary signals.

🤖 Our AI Solution: We've developed an intelligent system that can automatically analyze light curves from stars to detect the tiny dips in brightness that occur when a planet passes in front of its host star (called a transit).

⭐ Key Features:
• Advanced machine learning algorithms trained on NASA Kepler data
• Real-time visualization of planetary transits and solar systems
• Interactive data exploration and AI-powered predictions
• Feature importance analysis to understand what makes a planet detectable

🚀 Mission: Help astronomers discover new worlds and advance our understanding of planetary systems throughout the galaxy!"""
        
        desc_text.insert('1.0', problem_description)
        desc_text.configure(state='disabled')
        
        # Visualization section
        viz_frame = tk.Frame(scrollable_frame, bg='#0a0f1f')
        viz_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Solar System Visualization
        self.add_solar_system_viz(viz_frame)
        
        # Graph Interpreter Section
        self.add_graph_interpreter_viz(viz_frame)

    def add_solar_system_viz(self, parent):
        """Add embedded solar system visualization"""
        solar_frame = tk.LabelFrame(parent, 
                                   text="🌞 Exoplanet Transit Simulation",
                                   font=('Arial', 16, 'bold'),
                                   fg='#ffcc66',
                                   bg='#1a1f2f',
                                   labelanchor='n')
        solar_frame.pack(fill='x', pady=10)
        
        # Create embedded canvas for solar system
        self.solar_canvas = tk.Canvas(solar_frame, 
                                     width=600, 
                                     height=400, 
                                     bg='#0a0f1f',
                                     highlightthickness=0)
        self.solar_canvas.pack(pady=10)
        
        # Add control buttons
        control_frame = tk.Frame(solar_frame, bg='#1a1f2f')
        control_frame.pack(pady=10)
        
        start_btn = tk.Button(control_frame, 
                             text="▶️ Start Transit Animation",
                             command=self.start_solar_animation,
                             bg='#33cc33',
                             fg='white',
                             font=('Arial', 12, 'bold'))
        start_btn.pack(side='left', padx=10)
        
        stop_btn = tk.Button(control_frame,
                            text="⏹️ Stop Animation", 
                            command=self.stop_solar_animation,
                            bg='#cc3333',
                            fg='white',
                            font=('Arial', 12, 'bold'))
        stop_btn.pack(side='left', padx=10)
        
        # Initialize animation variables
        self.solar_animation_running = False
        self.solar_animation_thread = None

    def add_graph_interpreter_viz(self, parent):
        """Add graph interpreter visualization"""
        graph_frame = tk.LabelFrame(parent,
                                   text="📊 Light Curve Analysis",
                                   font=('Arial', 16, 'bold'),
                                   fg='#ffcc66', 
                                   bg='#1a1f2f',
                                   labelanchor='n')
        graph_frame.pack(fill='x', pady=10)
        
        # Create matplotlib figure for light curve
        self.light_curve_fig = Figure(figsize=(12, 4), facecolor='#1a1f2f')
        self.light_curve_ax = self.light_curve_fig.add_subplot(111, facecolor='#0a0f1f')
        self.light_curve_canvas = FigureCanvasTkAgg(self.light_curve_fig, master=graph_frame)
        self.light_curve_canvas.get_tk_widget().pack(fill='x', padx=10, pady=10)
        
        # Control panel for graph interpreter
        graph_control_frame = tk.Frame(graph_frame, bg='#1a1f2f')
        graph_control_frame.pack(pady=10)
        
        tk.Label(graph_control_frame, 
                text="🪐 Select Planet:",
                font=('Arial', 12, 'bold'),
                fg='white',
                bg='#1a1f2f').pack(side='left', padx=5)
        
        self.planet_var = tk.StringVar(value="Kepler-10b")
        planet_combo = ttk.Combobox(graph_control_frame, 
                                   textvariable=self.planet_var,
                                   values=["Kepler-10b", "Kepler-11b", "Kepler-22b", "Kepler-442b"],
                                   state="readonly",
                                   width=15)
        planet_combo.pack(side='left', padx=5)
        
        generate_btn = tk.Button(graph_control_frame,
                                text="📈 Generate Light Curve",
                                command=self.generate_light_curve,
                                bg='#66ccff',
                                fg='white',
                                font=('Arial', 11, 'bold'))
        generate_btn.pack(side='left', padx=10)

    def start_solar_animation(self):
        """Start the solar system animation"""
        if not self.solar_animation_running:
            self.solar_animation_running = True
            self.solar_animation_thread = threading.Thread(target=self.run_solar_animation, daemon=True)
            self.solar_animation_thread.start()

    def stop_solar_animation(self):
        """Stop the solar system animation"""
        self.solar_animation_running = False

    def run_solar_animation(self):
        """Run the solar system animation loop"""
        angle = 0
        star_x, star_y = 300, 200
        orbit_radius = 150
        planet_radius = 8
        star_radius = 20
        
        while self.solar_animation_running:
            try:
                # Clear canvas
                self.solar_canvas.delete("all")
                
                # Draw star
                self.solar_canvas.create_oval(star_x - star_radius, star_y - star_radius,
                                            star_x + star_radius, star_y + star_radius,
                                            fill='#ffcc66', outline='#ffe066', width=2)
                
                # Draw orbit
                self.solar_canvas.create_oval(star_x - orbit_radius, star_y - orbit_radius,
                                            star_x + orbit_radius, star_y + orbit_radius,
                                            outline='#334466', width=1)
                
                # Calculate planet position
                planet_x = star_x + orbit_radius * np.cos(angle)
                planet_y = star_y + orbit_radius * np.sin(angle)
                
                # Draw planet
                self.solar_canvas.create_oval(planet_x - planet_radius, planet_y - planet_radius,
                                            planet_x + planet_radius, planet_y + planet_radius,
                                            fill='#66ccff', outline='#88ccff', width=1)
                
                # Draw light rays when planet transits
                if abs(planet_x - star_x) < star_radius + planet_radius:
                    # Planet is transiting - draw blocked light
                    for i in range(50):
                        ray_angle = i * (2 * np.pi / 50)
                        ray_x = star_x + 250 * np.cos(ray_angle)
                        ray_y = star_y + 250 * np.sin(ray_angle)
                        
                        # Check if ray is blocked by planet
                        if np.sqrt((ray_x - planet_x)**2 + (ray_y - planet_y)**2) > planet_radius:
                            self.solar_canvas.create_line(star_x, star_y, ray_x, ray_y,
                                                         fill='#ffe066', width=1)
                else:
                    # Normal light rays
                    for i in range(50):
                        ray_angle = i * (2 * np.pi / 50)
                        ray_x = star_x + 250 * np.cos(ray_angle)
                        ray_y = star_y + 250 * np.sin(ray_angle)
                        self.solar_canvas.create_line(star_x, star_y, ray_x, ray_y,
                                                     fill='#ffe066', width=1)
                
                # Add labels
                self.solar_canvas.create_text(star_x, star_y + 50, text="⭐ Host Star",
                                            fill='#ffcc66', font=('Arial', 12, 'bold'))
                self.solar_canvas.create_text(planet_x, planet_y - 20, text="🪐 Exoplanet",
                                            fill='#66ccff', font=('Arial', 10, 'bold'))
                
                angle += 0.05
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Animation error: {e}")
                break

    def generate_light_curve(self):
        """Generate a sample light curve for the selected planet"""
        try:
            # Clear previous plot
            self.light_curve_ax.clear()
            
            # Generate sample data (simulating real exoplanet transit)
            time_days = np.linspace(0, 10, 1000)
            
            # Base stellar brightness with noise
            baseline_flux = 1.0
            noise = np.random.normal(0, 0.001, len(time_days))
            flux = np.ones_like(time_days) + noise
            
            # Add transit dips
            transit_period = 3.5  # days
            transit_duration = 0.15  # days
            transit_depth = 0.01  # 1% dip
            
            for i, t in enumerate(time_days):
                # Check if we're in a transit
                phase = (t % transit_period) / transit_period
                if abs(phase - 0.5) < (transit_duration / transit_period / 2):
                    # Create transit shape (simplified)
                    transit_phase = (phase - 0.5) / (transit_duration / transit_period / 2)
                    if abs(transit_phase) < 1:
                        flux[i] -= transit_depth * (1 - transit_phase**2)
            
            # Plot the light curve
            self.light_curve_ax.plot(time_days, flux, 'w-', linewidth=1, alpha=0.7)
            self.light_curve_ax.scatter(time_days[::50], flux[::50], c='#66ccff', s=8, alpha=0.8)
            
            # Highlight transit regions
            for t_start in np.arange(0, 10, transit_period):
                transit_center = t_start + transit_period/2
                if transit_center < 10:
                    self.light_curve_ax.axvspan(transit_center - transit_duration/2, 
                                              transit_center + transit_duration/2,
                                              alpha=0.3, color='red', label='Transit' if t_start == 0 else "")
            
            self.light_curve_ax.set_xlabel('Time (days)', color='white', fontsize=12)
            self.light_curve_ax.set_ylabel('Relative Flux', color='white', fontsize=12)
            self.light_curve_ax.set_title(f'🌟 Light Curve for {self.planet_var.get()} - Transit Detection',
                                         color='#ffcc66', fontsize=14, fontweight='bold')
            
            # Style the plot
            self.light_curve_ax.set_facecolor('#0a0f1f')
            self.light_curve_ax.tick_params(colors='white')
            self.light_curve_ax.spines['bottom'].set_color('white')
            self.light_curve_ax.spines['top'].set_color('white') 
            self.light_curve_ax.spines['right'].set_color('white')
            self.light_curve_ax.spines['left'].set_color('white')
            
            if len(self.light_curve_ax.get_legend_handles_labels()[0]) > 0:
                legend = self.light_curve_ax.legend(facecolor='#1a1f2f', edgecolor='white')
                legend.get_texts()[0].set_color('white')
            
            self.light_curve_fig.patch.set_facecolor('#1a1f2f')
            self.light_curve_canvas.draw()
            
        except Exception as e:
            print(f"Error generating light curve: {e}")

    def build_ui(self):
        """Build the main UI with tabs"""
        self.root.geometry('1400x900')
        
        # Play background music
        self.play_background_music()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.home_tab = tk.Frame(self.notebook, bg='#0a0f1f')
        self.data_tab = ttk.Frame(self.notebook)
        self.ml_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.home_tab, text="🏠 Home")
        self.notebook.add(self.data_tab, text="📊 Data Explorer")
        self.notebook.add(self.ml_tab, text="🤖 AI Predictions")
        
        # Build home tab
        self.build_home_tab()
        
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
