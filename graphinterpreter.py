import csv
import random
import tkinter as tk

# --- Prompt user ---
planet_input = input("Enter kepler_name or kepoi_name of the planet: ").strip()

# --- Load data from CSV ---
csv_file = "keplerdataset.csv"
planet_data = None
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['kepler_name'].strip() == planet_input or row['kepoi_name'].strip() == planet_input:
            planet_data = row
            break

if planet_data is None:
    print(f"Planet '{planet_input}' not found in dataset.")
    exit()

# --- PARAMETERS FROM CSV ---
period = float(planet_data['koi_period'])             # days
duration = float(planet_data['koi_duration']) / 24.0  # hours → days
depth = float(planet_data['koi_depth']) / 1e6         # ppm → normalized flux
cadence = 10 / (60*24)                                 # 10 minutes in days
total_days = 3 * period
noise_level = 0.0002

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

# --- PLOT LIGHT CURVE USING TKINTER (rest of your code unchanged) ---
WIDTH = 900
HEIGHT = 500
MARGIN = 80
min_flux = min(flux)
max_flux = max(flux)
min_time = min(time)
max_time = max(time)

root = tk.Tk()
root.title(f"{planet_input} Light Curve")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

# Draw axes
canvas.create_line(MARGIN, HEIGHT-MARGIN, WIDTH-MARGIN, HEIGHT-MARGIN, width=2)
canvas.create_line(MARGIN, MARGIN, MARGIN, HEIGHT-MARGIN, width=2)
canvas.create_text(WIDTH/2, HEIGHT-30, text="Time [days]", font=("Arial", 12))
canvas.create_text(40, HEIGHT/2, text="Normalized Flux", angle=90, font=("Arial", 12))

# X-axis ticks
num_x_ticks = 6
for i in range(num_x_ticks + 1):
    x_val = min_time + i * (max_time - min_time) / num_x_ticks
    x = MARGIN + (x_val - min_time) / (max_time - min_time) * (WIDTH - 2*MARGIN)
    canvas.create_line(x, HEIGHT-MARGIN, x, HEIGHT-MARGIN+10, width=2)
    canvas.create_text(x, HEIGHT-MARGIN+25, text=f"{x_val:.1f}", font=("Arial", 10))

# Y-axis ticks
num_y_ticks = 5
for i in range(num_y_ticks + 1):
    f_val = min_flux + i * (max_flux - min_flux) / num_y_ticks
    y = HEIGHT - MARGIN - (f_val - min_flux) / (max_flux - min_flux) * (HEIGHT - 2*MARGIN)
    canvas.create_line(MARGIN-10, y, MARGIN, y, width=2)
    canvas.create_text(MARGIN-40, y, text=f"{f_val:.6f}", font=("Arial", 10))

# Plot light curve
prev_x, prev_y = None, None
for t_val, f_val in zip(time, flux):
    x = MARGIN + (t_val - min_time) / (max_time - min_time) * (WIDTH - 2*MARGIN)
    y = HEIGHT - MARGIN - (f_val - min_flux) / (max_flux - min_flux) * (HEIGHT - 2*MARGIN)
    phase = t_val % period
    color = "red" if 0 <= phase <= duration else "blue"
    if prev_x is not None:
        canvas.create_line(prev_x, prev_y, x, y, fill=color)
    prev_x, prev_y = x, y

root.mainloop()
