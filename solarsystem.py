import tkinter as tk
import math
import time
         

# --- PARAMETERS ---
orbit_radius = 200
planet_radius = 12
star_radius = 25
frame_rate = 0.02
num_rays = 300
ray_length = 280
sensor_height = 40
sensor_width = 100

# --- COLORS ---
bg_color = "#0a0f1f"
star_color = "#ffcc66"
planet_color = "#66ccff"
orbit_color = "#334466"
ray_color = "#ffe066"
bar_color = "#33cc33"

# --- CANVAS SETUP ---
WIDTH = 600
HEIGHT = 750
root = tk.Tk()
root.title("Exoplanet Transit Visualization – Full Ray Cutoff")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=bg_color, highlightthickness=0)
canvas.pack()

# --- RANDOM STARS BACKGROUND ---
import random
num_stars = 50  # number of stars
for _ in range(num_stars):
    sx = random.randint(0, WIDTH)
    sy = random.randint(0, HEIGHT)  # only upper half
    size = random.randint(1, 3)
    canvas.create_oval(sx, sy, sx + size, sy + size, fill="white", outline="")

cx, cy = WIDTH / 2, HEIGHT / 2 - 40  # star center

# --- DRAW LIGHT RAYS (stationary) ---
rays = []
for i in range(num_rays):
    angle = 2 * math.pi * i / num_rays
    x_end = cx + ray_length * math.cos(angle)
    y_end = cy + ray_length * math.sin(angle)
    r = canvas.create_line(cx, cy, x_end, y_end, fill=ray_color, width=1)
    rays.append((r, angle))

# --- DRAW STAR AND ORBIT ---
star = canvas.create_oval(cx - star_radius, cy - star_radius,
                          cx + star_radius, cy + star_radius,
                          fill=star_color, outline="")
canvas.create_oval(cx - orbit_radius, cy - orbit_radius,
                   cx + orbit_radius, cy + orbit_radius,
                   outline=orbit_color, width=2)

# --- CREATE PLANET ---
planet = canvas.create_oval(0, 0, 0, 0, fill=planet_color, outline="")

# --- TELESCOPE (stylized) ---
scope_base_y = HEIGHT - 50
scope_width = 80
scope_height = 40
scope_x1 = cx - scope_width / 2
scope_x2 = cx + scope_width / 2

# Base (tripod-like support)
canvas.create_polygon(cx - 10, scope_base_y + 10,
                      cx + 10, scope_base_y + 10,
                      cx, scope_base_y + 25,
                      fill="#444")

# Telescope tube
telescope = canvas.create_rectangle(scope_x1, scope_base_y - scope_height,
                                    scope_x2, scope_base_y,
                                    fill="#888", outline="#555")

# Telescope lens (opening)
canvas.create_oval(cx - 20, scope_base_y - scope_height - 10,
                   cx + 20, scope_base_y - scope_height + 10,
                   fill="#333", outline="#666")

# Telescope coordinates (for later reference)
tel_x = cx
tel_y = scope_base_y - scope_height / 2

# --- BRIGHTNESS INDICATOR BAR ---
bar_x0 = tel_x - 150
bar_y0 = tel_y + 60
bar_x1 = tel_x + 150
bar_y1 = bar_y0 + 20
canvas.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, outline="white")
brightness_bar = canvas.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, fill=bar_color, width=0)

# Axis labels (0, 0.2, … 1.0)
for i in range(6):
    x = bar_x0 + i * (bar_x1 - bar_x0) / 5
    canvas.create_text(x, bar_y1 + 12, text=f"{i/5:.1f}", fill="white", font=("Arial", 10))

# --- DEFINE DETECTION RAYS (16 total now) ---
center_angle = math.pi / 2  # downward direction
angle_spread = math.pi / 12  # ~15 degrees
detection_rays = []
for i in range(22):  # 16 -> 22 (added 3 more each side)
    a = center_angle - angle_spread / 2 + (angle_spread * i / 21)
    closest_ray = min(rays, key=lambda r: abs((r[1] - a + math.pi) % (2 * math.pi) - math.pi))
    detection_rays.append(closest_ray)

# --- ANIMATION LOOP ---
angle = 0.0
while True:
    # Planet position
    x = cx + orbit_radius * math.cos(angle)
    y = cy + orbit_radius * math.sin(angle)
    canvas.coords(planet, x - planet_radius, y - planet_radius, x + planet_radius, y + planet_radius)

    blocked_count = 0  # how many detection rays are blocked

    # Check *all* rays for visual blocking
    for ray, ray_angle in rays:
        rx_end = cx + ray_length * math.cos(ray_angle)
        ry_end = cy + ray_length * math.sin(ray_angle)
        dx, dy = rx_end - cx, ry_end - cy
        t = ((x - cx) * dx + (y - cy) * dy) / (dx * dx + dy * dy)
        closest_x = cx + t * dx
        closest_y = cy + t * dy
        dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        if dist < planet_radius and 0 <= t <= 1:
            # shorten visually
            ray_length_cut = max(0, (math.sqrt((x - cx)**2 + (y - cy)**2) - planet_radius))
            cut_x = cx + ray_length_cut * math.cos(ray_angle)
            cut_y = cy + ray_length_cut * math.sin(ray_angle)
            canvas.coords(ray, cx, cy, cut_x, cut_y)
        else:
            # restore full length
            canvas.coords(ray, cx, cy, rx_end, ry_end)

    # Now calculate detector brightness based only on its designated rays
    for ray, ray_angle in detection_rays:
        rx_end = cx + ray_length * math.cos(ray_angle)
        ry_end = cy + ray_length * math.sin(ray_angle)
        dx, dy = rx_end - cx, ry_end - cy
        t = ((x - cx) * dx + (y - cy) * dy) / (dx * dx + dy * dy)
        closest_x = cx + t * dx
        closest_y = cy + t * dy
        dist = math.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        if dist < planet_radius and 0 <= t <= 1:
            blocked_count += 1

    # Brightness drops by 1/22 per blocked ray
    brightness_ratio = max(0.0, 1.0 - blocked_count / 22.0)

    # Update brightness bar
    bar_fill_width = bar_x0 + brightness_ratio * (bar_x1 - bar_x0)
    canvas.coords(brightness_bar, bar_x0, bar_y0, bar_fill_width, bar_y1)

    # Dim telescope color proportionally
    dim_intensity = int(100 + 155 * brightness_ratio)
    dim_color = f"#{dim_intensity:02x}{dim_intensity:02x}ff"
    canvas.itemconfig(telescope, fill=dim_color)

    root.update()
    time.sleep(frame_rate)
    angle += 0.03
