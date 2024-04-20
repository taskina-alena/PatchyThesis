#functions to draw patchy particles

import matplotlib.pyplot as plt
import numpy as np

# Function to draw a circle
def draw_circle(center, radius, ax, color, fill=False):
    circle = plt.Circle(center, radius, color=color, fill=fill)
    ax.add_patch(circle)

# Function to draw a patchy particle with 3 patches
def draw_particle(central_center, rotation, ax, central_radius=0.5, small_radius=0.12, color='#56b391', fill=False, color_p = '#c986ff', fill_p = True):
    draw_circle(central_center, central_radius, ax, color=color, fill=fill)
    angles = np.deg2rad([0+rotation, 0+rotation+120, 0+rotation+240])
    for angle in angles:
        x = central_center[0] + central_radius * np.cos(angle)
        y = central_center[1] + central_radius * np.sin(angle)
        draw_circle((x, y), small_radius, ax, color_p, fill_p)

# Function to draw a line and show its length
def draw_line(point1, point2, length, ax, display_l=True, linestyle='--', color='k', fontsize=15):
    # Draw the line
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color=color, linestyle=linestyle)
    # Display the length
    if display_l:
        ax.text(point2[0], point2[1], length, fontsize=15)
