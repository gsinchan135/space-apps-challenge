import random
import tkinter as tk

# --- PARAMETERS (from your CSV) ---
period = 9.48803557            # days
duration = 2.9575 / 24         # hours -> days
depth = 615.8 / 1e6            # ppm -> normalized flux
cadence = 10 / (60*24)         # 10 minutes in days
total_days = 3 * period        # simulate 3 orbital periods
noise_level = 0.0002           # small noise

# --- GENERATE TIME AND FLUX DATA ---
time = []
flux = []

t = 0.0
while t <= total_days:
    phase = t % period
    if 0 <= phase <= duration:
        f = 1.0 - depth
    else:
        f = 1.0
    f += random.uniform(-noise_level, noise_level)
    time.append(t)
    flux.append(f)
    t += cadence

# --- PLOT LIGHT CURVE USING TKINTER ---
WIDTH = 900
HEIGHT = 500
MARGIN = 80  # bigger margin for numbers

min_flux = min(flux)
max_flux = max(flux)
min_time = min(time)
max_time = max(time)

root = tk.Tk()
root.title("Kepler-227 b Light Curve (Simulated)")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

# Draw axes
canvas.create_line(MARGIN, HEIGHT-MARGIN, WIDTH-MARGIN, HEIGHT-MARGIN, width=2)  # x-axis
canvas.create_line(MARGIN, MARGIN, MARGIN, HEIGHT-MARGIN, width=2)                # y-axis

# Axis labels
canvas.create_text(WIDTH/2, HEIGHT-30, text="Time [days]", font=("Arial", 12))
canvas.create_text(40, HEIGHT/2, text="Normalized Flux", angle=90, font=("Arial", 12))

# --- Draw X-axis ticks and numbers ---
num_x_ticks = 6
for i in range(num_x_ticks + 1):
    x_val = min_time + i * (max_time - min_time) / num_x_ticks
    x = MARGIN + (x_val - min_time) / (max_time - min_time) * (WIDTH - 2*MARGIN)
    # Tick mark
    canvas.create_line(x, HEIGHT-MARGIN, x, HEIGHT-MARGIN+10, width=2)
    # Number
    canvas.create_text(x, HEIGHT-MARGIN+25, text=f"{x_val:.1f}", font=("Arial", 10))

# --- Draw Y-axis ticks and numbers ---
num_y_ticks = 5
for i in range(num_y_ticks + 1):
    f_val = min_flux + i * (max_flux - min_flux) / num_y_ticks
    y = HEIGHT - MARGIN - (f_val - min_flux) / (max_flux - min_flux) * (HEIGHT - 2*MARGIN)
    # Tick mark
    canvas.create_line(MARGIN-10, y, MARGIN, y, width=2)
    # Number
    canvas.create_text(MARGIN-40, y, text=f"{f_val:.6f}", font=("Arial", 10))

# --- Plot light curve and highlight transits ---
prev_x, prev_y = None, None
for t_val, f_val in zip(time, flux):
    x = MARGIN + (t_val - min_time) / (max_time - min_time) * (WIDTH - 2*MARGIN)
    y = HEIGHT - MARGIN - (f_val - min_flux) / (max_flux - min_flux) * (HEIGHT - 2*MARGIN)
    
    # Highlight transit dips in red
    phase = t_val % period
    if 0 <= phase <= duration:
        color = "red"
    else:
        color = "blue"
    
    if prev_x is not None:
        canvas.create_line(prev_x, prev_y, x, y, fill=color)
    prev_x, prev_y = x, y

root.mainloop()
