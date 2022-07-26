Unsupervised metrics|Description
:---|:---
collection|Name of collection to evaluate.
traj_count|Total number trajectory documents.
timestamp_count|Total number of unique timestamp (resampled at 25Hz).
duration|Length of the collection in sec.
x_traveled|xmax - xmin for each trajectory.
y_traveled|ymax - ymin for each trajectory.
max_vx|Max finite-difference of x over finite-difference of timestamp of each trajectory.
min_vx|Min finite-difference of x over finite-difference of timestamp of each trajectory.
max_vy|Max finite-difference of y over finite-difference of timestamp of each trajectory.
min_vy|Min finite-difference of y over finite-difference of timestamp of each trajectory.
max_ax|Max second-order finite-difference of x over finite-difference of timestamp of each trajectory.
min_ax|Min second-order finite-difference of x over finite-difference of timestamp of each trajectory.
avg_vx|Avg finite-difference of x over finite-difference of timestamp of each trajectory.
avg_vy|Avg finite-difference of y over finite-difference of timestamp of each trajectory.
lane_changes|Number of lane changes of each trajectory.
backward_cars|Trajectory IDs whose finite-difference of x is negative at any instance.
min_spacing|Min difference in x of all car-following pairs (at the same lane) at a given timestamp.
overlaps|Trajectory pairs whose bboxes overlap at a given timestamp.


