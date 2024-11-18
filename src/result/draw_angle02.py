import json
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the JSON data from the file
file_path = 'trajectories_00.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract the 5th trajectory from the data
trajectory_data = data[3]

# Extract key points from the 5th trajectory
throw_point_data = trajectory_data['key_points']['throw_point']
highest_point_data = trajectory_data['key_points']['highest_point']

# Extract coordinates of throw point and highest point
throw_point = np.array([throw_point_data['x'], throw_point_data['y'], throw_point_data['z']])
highest_point = np.array([highest_point_data['x'], highest_point_data['y'], highest_point_data['z']])

# Vector from throw point to highest point
line_vector = highest_point - throw_point

# Vector N: vertical normal vector at the throw point (assumed as z-axis)
N = np.array([0, 0, 1])  # assuming a vertical direction in the z-axis

# Calculate the cosine of the angle between line vector and N
cos_theta = line_vector[2] / np.linalg.norm(line_vector)

# Calculate the angle in radians, then convert to degrees
theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
theta_degrees = math.degrees(theta_radians)

# Print the angle
print(f"The angle θ between the vertical normal vector and the line connecting the throw and highest point is {theta_degrees:.2f} degrees")

# Plot the trajectory and the key points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory points
trajectory = trajectory_data['trajectory']
trajectory_x = [-point['x'] for point in trajectory]  # Reverse x-axis
trajectory_y = [-point['y'] for point in trajectory]  # Reverse y-axis
trajectory_z = [point['z'] for point in trajectory]

ax.plot(trajectory_x, trajectory_y, trajectory_z, color='black')

# Plot throw point and highest point
throw_point[0] = -throw_point[0]  # Reverse x-axis of throw point
throw_point[1] = -throw_point[1]  # Reverse y-axis of throw point
highest_point[0] = -highest_point[0]  # Reverse x-axis of highest point
highest_point[1] = -highest_point[1]  # Reverse y-axis of highest point

#ax.scatter(*throw_point, color='yellow')
#ax.scatter(*highest_point, color='red')

# Plot vertical normal vector N from throw point
#ax.quiver(*throw_point, N[0], N[1], N[2], color='purple', label='Vertical Normal Vector N',linestyle='--')

# Draw a line connecting throw point and highest point
#ax.plot([throw_point[0], highest_point[0]], [throw_point[1], highest_point[1]], [throw_point[2], highest_point[2]], color='purple', linestyle='--', label='Line Connecting Points')

# Annotate the angle on the plot
#mid_point = (throw_point + highest_point) / 2
#ax.text(mid_point[0]-0.008, mid_point[1] - 0.05, mid_point[2], f"θ", color='purple' , fontsize=20)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show the plot
plt.show()
