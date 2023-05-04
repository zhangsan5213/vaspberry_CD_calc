import os
import re
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def read_kpath_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_points_per_path = int(lines[1].split()[0])
    kpoints = []
    labels = []
    start = False
    for line in lines:
        if "Reciprocal" in line:
            start = True
            continue
        if start:
            if len(line.strip()) == 0:
                continue
            kpoints.append([float(x) for x in line.split()[:3]])
            labels.append(line.split()[3])

    kpoint_pairs = [(kpoints[i], kpoints[i + 1]) for i in range(0, len(kpoints), 2)]
    label_pairs = [(labels[i], labels[i + 1]) for i in range(0, len(labels), 2)]

    return n_points_per_path, kpoint_pairs, label_pairs

def interpolate_kpoints(kpoint_pairs, n_points):
    interpolated_points = []
    for pair in kpoint_pairs:
        start, end = pair
        start, end = np.array(start), np.array(end)
        points = [start + (end - start) * t for t in np.linspace(0, 1, n_points)]
        interpolated_points.extend(points)
    return np.array(interpolated_points)

def read_second_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    cart_kpoints = []
    selectivities = []
    start = False
    for line in lines:
        if "(cart) kx" in line:
            start = True
            continue
        if start:
            if len(line.strip()) == 0:
                continue
            data = [float(x) for x in line.split()]
            cart_kpoints.append(data[:3])
            selectivities.append(data[3])

    return cart_kpoints, selectivities

def find_nearest_point(cart_kpoint, interpolated_points, threshold=0.1):
    min_distance = float('inf')
    nearest_point = None
    for point in interpolated_points:
        distance = np.linalg.norm(np.array(cart_kpoint) - np.array(point))
        if distance < min_distance:
            min_distance = distance
            nearest_point = point

    if min_distance <= threshold:
        return nearest_point, min_distance
    else:
        return None, None

def read_band_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    band_data = {}
    band_index = None
    for line in lines:
        if "# Band-Index" in line:
            band_index = int(line.split()[-1])
            band_data[band_index] = []
        elif (band_index is not None) and (line != " \n"):
            kpath, energy = [float(x) for x in line.split()]
            band_data[band_index].append((kpath, energy))

    return band_data

def extract_band_indices_from_filename(filename):
    regex = r"CIRC_DICHROISM\.CD_(\d+)_(\d+)\.dat"
    match = re.search(regex, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError("Invalid filename format")

def find_energy_difference(nearest_point, band_data, initial_band, final_band):
    initial_energy = None
    final_energy = None

    for kpath, energy in band_data[initial_band]:
        if np.isclose(kpath, nearest_point[0], rtol=1e-5):
            initial_energy = energy
            break

    for kpath, energy in band_data[final_band]:
        if np.isclose(kpath, nearest_point[0], rtol=1e-5):
            final_energy = energy
            break

    if initial_energy is not None and final_energy is not None:
        return final_energy - initial_energy
    else:
        return None

## MAIN ##

filename = 'KPOINTS'  # Replace with your filename
n_points_per_path, kpoint_pairs, label_pairs = read_kpath_file(filename)
interpolated_points = interpolate_kpoints(kpoint_pairs, n_points_per_path)

# print("Total number of interpolated points:", len(interpolated_points))
# print("Interpolated points:\n", interpolated_points)

# filename2 = './R/CIRC_DICHROISM.CD_236_241.dat.dat'  # Replace with your second file's name
# cart_kpoints, selectivities = read_second_file(filename2)

filename3 = 'BAND_R_SOC.dat'  # Replace with your BAND.dat file's name
band_data = read_band_file(filename3)

second_file_names = ["./R/"+i for i in os.listdir("./R/")]

transition_selectivities, transition_energy_diffs = [], []
for second_file_name in tqdm(second_file_names, ncols=50):
    cart_kpoints, selectivities = read_second_file(second_file_name)
    initial_band, final_band = extract_band_indices_from_filename(second_file_name)
    
    this_transition_selectivity, this_energy_diffs = [], []
    for i, cart_kpoint in enumerate(cart_kpoints):
        nearest_point, distance = find_nearest_point(cart_kpoint, interpolated_points)
        if nearest_point is not None:
            energy_diff = find_energy_difference(nearest_point, band_data, initial_band, final_band)
            if energy_diff is not None:
                # print(f"Cartesian point: {cart_kpoint}, Selectivity: {selectivities[i]:.6f}, Nearest point: {nearest_point}, Energy difference: {energy_diff:.6f}")
                # print(f"Selectivity: {selectivities[i]:.6f}, Energy difference: {energy_diff:.6f}")
                this_transition_selectivity.append(selectivities[i])
                this_energy_diffs.append(energy_diff)

    transition_selectivities.append(sum(this_transition_selectivity))
    transition_energy_diffs.append(sum(this_energy_diffs)/len(this_energy_diffs))

    # for cart_kpoint in cart_kpoints:
    #     nearest_point, distance = find_nearest_point(cart_kpoint, interpolated_points)
    #     if nearest_point is not None:
    #         energy_diff = find_energy_difference(nearest_point, band_data, initial_band, final_band)
    #         if energy_diff is not None:
    #             print(f"Cartesian point: {cart_kpoint}, Nearest point: {nearest_point}, Energy difference: {energy_diff:.6f}")
    #     #     else:
    #     #         print(f"Cartesian point: {cart_kpoint}, Nearest point: {nearest_point}, Energy difference: Not found")
    #     # else:
    #     #     print(f"Cartesian point: {cart_kpoint} is discarded")


# Define the Gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Create the figure and axis objects
fig, ax = plt.subplots()

# Loop through each x and y value
x_vals = np.linspace(min(transition_energy_diffs), max(transition_energy_diffs), 100)
for i, xi_yi in enumerate(zip(transition_energy_diffs, transition_selectivities)):
    # Extract the dtaa
    xi, yi = xi_yi
    # Calculate the Gaussian width
    width = 0.1
    # Calculate the Gaussian function values
    if i == 0:
        y_vals = gaussian(x_vals, yi, xi, width)
    else:
        y_vals+= gaussian(x_vals, yi, xi, width)

# Plot the Gaussian curve
ax.plot(1240/x_vals, y_vals, 'b-', alpha=0.5)
ax.plot(1240/x_vals,-y_vals, 'r-', alpha=0.5)

# Add labels and title
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('CD (a.u.)')
ax.set_title('Circular Dichroism plot')

# Show the plot
plt.show()

with open('20230504_CD.txt', 'w') as f:
    for x,y in zip(1240/x_vals, y_vals):
        f.write(str(x) + '\t' + str(y) + '\n')