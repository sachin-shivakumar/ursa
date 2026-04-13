import os
import h5py
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import pickle
import pandas as pd
from sklearn.preprocessing import quantile_transform


#from collections import defaultdict
import networkx as nx

#cond activate plot_hdf5

# Your original plotting function
# def plot_damage_lines_within_range(data, minValue=0.25, maxValue=2, alpha=0.05):
#     moddata = data.copy()
#     x1, y1 = moddata[:, 0], moddata[:, 1]
#     x2, y2 = moddata[:, 3], moddata[:, 4]
#     mask = (x1 > 0.001) & (x1 < 0.248) & (y1 > 0.001) & (y1 < 0.25) & \
#            (x2 > 0.001) & (x2 < 0.248) & (y2 > 0.001) & (y2 < 0.25)
#     moddata = moddata[mask]
#     damage = moddata[:, 6].copy()
#     damage[damage == 2.0] = 1.0
#     for i in range(len(moddata)):
#         if damage[i] > minValue:
#             plt.plot([moddata[i, 0], moddata[i, 3]], [moddata[i, 1], moddata[i, 4]],
#                      color=plt.cm.hot_r(damage[i]), alpha=alpha)

def plot_damage_lines_within_range(data, minValue=0.25, maxValue=2, alpha=0.05, ax=None):
    """
    Plot crack line segments with damage values in a given range.

    Parameters
    ----------
    data : np.ndarray
        Crack data array (expects at least 7 columns).
    minValue : float
        Minimum damage threshold.
    maxValue : float
        Maximum damage threshold.
    alpha : float
        Transparency of lines.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new one.
    """
    if ax is None:
        fig, ax = plt.subplots()

    moddata = data.copy()

    # Coordinates
    x1, y1 = moddata[:, 0], moddata[:, 1]
    x2, y2 = moddata[:, 3], moddata[:, 4]

    # Filter spatial domain
    mask = (
        (x1 > 0.001) & (x1 < 0.248) &
        (y1 > 0.001) & (y1 < 0.25) &
        (x2 > 0.001) & (x2 < 0.248) &
        (y2 > 0.001) & (y2 < 0.25)
    )
    moddata = moddata[mask]

    # Damage values
    damage = moddata[:, 6].copy()
    damage[damage == 2.0] = 1.0  # treat "2.0" as fully broken

    # Plot each line if above threshold
    for i in range(len(moddata)):
        if minValue < damage[i] <= maxValue:
            ax.plot(
                [moddata[i, 0], moddata[i, 3]],
                [moddata[i, 1], moddata[i, 4]],
                color=plt.cm.hot_r(damage[i]),
                alpha=alpha
            )

    # Set equal aspect + limits
    ax.set_aspect("equal")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return ax

def plot_damage_lines_with_max_path(crack_data, max_path, minValue=0.25, maxValue=2, alpha=1.0):
    moddata = crack_data.copy()
    x1, y1 = moddata[:, 0], moddata[:, 1]
    x2, y2 = moddata[:, 3], moddata[:, 4]

    # Spatial mask to exclude boundary artifacts
    mask = (x1 > 0.001) & (x1 < 0.248) & (y1 > 0.001) & (y1 < 0.25) & \
           (x2 > 0.001) & (x2 < 0.248) & (y2 > 0.001) & (y2 < 0.25)
    moddata = moddata[mask]

    # Damage filtering
    damage = moddata[:, 6].copy()
    damage[damage == 2.0] = 1.0

    # Plot base cracks
    for i in range(len(moddata)):
        if damage[i] > minValue:
            plt.plot([moddata[i, 0], moddata[i, 3]],
                     [moddata[i, 1], moddata[i, 4]],
                     color=plt.cm.Greys(damage[i] / maxValue), alpha=alpha)

    # Plot the max path in red
    # Plot red lines between each pair of consecutive nodes in the max_path
    for i in range(len(max_path) - 1):
        x_vals = [max_path[i][0], max_path[i+1][0]]
        y_vals = [max_path[i][1], max_path[i+1][1]]
        plt.plot(x_vals, y_vals, color='red', linewidth=2.5, zorder=10)

    # Optional: highlight endpoints
    plt.scatter([max_path[0][0], max_path[-1][0]],
                [max_path[0][1], max_path[-1][1]],
                color='red', s=20, zorder=11)

    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Crack Network with Longest Connected Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

def find_longest_path_between_nodes(graph, start, end):
    """
    Find the longest simple path between two nodes using DFS.
    Returns (path, total_distance)
    """
    def dfs(current, target, visited, current_path, current_distance):
        if current == target:
            return current_path.copy(), current_distance
        
        best_path = []
        best_distance = 0.0
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                edge_weight = graph[current][neighbor]['weight']
                visited.add(neighbor)
                current_path.append(neighbor)
                
                path, distance = dfs(neighbor, target, visited, current_path, 
                                   current_distance + edge_weight)
                
                if distance > best_distance:
                    best_distance = distance
                    best_path = path.copy()
                
                # Backtrack
                current_path.pop()
                visited.remove(neighbor)
        
        return best_path, best_distance
    
    visited = {start}
    path, distance = dfs(start, end, visited, [start], 0.0)
    return path, distance

def snap_point(p, tol=1e-3):
    return (round(p[0]/tol)*tol, round(p[1]/tol)*tol)

def find_longest_connected_crack(crack_data):
    G = nx.Graph()
    for row in crack_data:
        p1 = snap_point((row[0], row[1]))
        p2 = snap_point((row[3], row[4]))
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        G.add_edge(p1, p2, weight=dist)

    max_distance = 0.0
    max_path = []

    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)

        # Get both distances and paths
        # Method 1: build dicts from the generator
        lengths = {}
        paths = {}
        for source, (dist_dict, path_dict) in nx.all_pairs_dijkstra(subgraph, weight="weight"):
            lengths[source] = dist_dict
            paths[source] = path_dict

        for u in lengths:
            for v in lengths[u]:
                dist = lengths[u][v]
                if dist > max_distance:
                    max_distance = dist
                    max_path = paths[u][v]  # the actual node sequence

    #print("Longest path:", max_path)
    #plot_damage_lines_with_max_path(crack_data, max_path, minValue=0.9)
    # print(max_distance)
    # print(max_path)
    return max_distance, max_path

def calculate_crack_metrics(data, data_initial, damage_threshold_min=0.25, damage_threshold_max=2.0):
    """
    Calculate crack metrics for a given time step data
    
    Parameters:
    data: numpy array with crack data
    data_initial: numpy array with initial crack data
    damage_threshold_min: minimum damage value to consider as crack
    damage_threshold_max: maximum damage value to consider as crack
    
    Returns:
    dict with metrics:
        total_cracks, crack_density, max_euclidean_distance,
        num_clusters, largest_cluster_fraction, avg_orientation, orientation_std
    """
    if len(data) == 0:
        return {
            'total_cracks': 0,
            'crack_density': 0.0,
            'max_euclidean_distance': 0.0,
            'num_clusters': 0,
            'largest_cluster_fraction': 0.0,
            'avg_orientation': 0.0,
            'orientation_std': 0.0,
        }
    
    # --- Spatial filtering ---
    x1, y1 = data[:, 0], data[:, 1]
    x2, y2 = data[:, 3], data[:, 4]
    mask = (x1 > 0.001) & (x1 < 0.248) & (y1 > 0.001) & (y1 < 0.25) & \
           (x2 > 0.001) & (x2 < 0.248) & (y2 > 0.001) & (y2 < 0.25)
    filtered_data = data[mask]
    
    if len(filtered_data) == 0:
        return {
            'total_cracks': 0,
            'crack_density': 0.0,
            'max_euclidean_distance': 0.0,
            'num_clusters': 0,
            'largest_cluster_fraction': 0.0,
            'avg_orientation': 0.0,
            'orientation_std': 0.0,
        }

    # --- Initial fracture filtering ---
    x1i, y1i = data_initial[:, 0], data_initial[:, 1]
    x2i, y2i = data_initial[:, 3], data_initial[:, 4]
    mask_init = (x1i > 0.001) & (x1i < 0.248) & (y1i > 0.001) & (y1i < 0.25) & \
                (x2i > 0.001) & (x2i < 0.248) & (y2i > 0.001) & (y2i < 0.25)
    filtered_data_initial = data_initial[mask_init]

    initial_fracs = np.sum(filtered_data_initial[:, 6] == 2.0)

    # --- Damage filtering ---
    damage = filtered_data[:, 6]
    crack_mask = (damage > damage_threshold_min) & (damage <= damage_threshold_max)
    crack_data = filtered_data[crack_mask]

    if len(crack_data) < initial_fracs:
        raise ValueError('initial fracs is greater than the evolved cracks')

    # --- 1. Total cracks (normalized) ---
    total_cracks = (len(crack_data) - initial_fracs) / len(filtered_data)

    # --- 2. Crack density ---
    domain_area = 0.25 * 0.25
    crack_density = total_cracks / domain_area

    # --- 3. Max euclidean distance along connected cracks ---
    max_euclidean_distance = 0.0
    if len(crack_data) > 0:
        max_euclidean_distance, _ = find_longest_connected_crack(crack_data)

    # --- 4. Connectivity / cluster analysis ---
    G = nx.Graph()
    for row in crack_data:
        p1 = (row[0], row[1])
        p2 = (row[3], row[4])
        G.add_edge(p1, p2)

    num_clusters = nx.number_connected_components(G)
    largest_cluster_fraction = 0.0
    if num_clusters > 0:
        largest_cluster_size = max(len(c) for c in nx.connected_components(G))
        largest_cluster_fraction = largest_cluster_size / len(crack_data)

    # --- 5. Crack orientation statistics ---
    # angle of each crack segment w.r.t horizontal
    angles = np.arctan2(crack_data[:, 4] - crack_data[:, 1], crack_data[:, 3] - crack_data[:, 0])
    avg_orientation = np.mean(angles)
    orientation_std = np.std(angles)

    return {
        'total_cracks': total_cracks,
        'crack_density': crack_density,
        'max_euclidean_distance': max_euclidean_distance,
        'num_clusters': num_clusters,
        'largest_cluster_fraction': largest_cluster_fraction,
        'avg_orientation': avg_orientation,
        'orientation_std': orientation_std,
    }

def extract_time_step_datasets(h5_file, fracture_num):
    """
    Extract all time step datasets from an HDF5 file for a given fracture number
    
    Returns:
    list of tuples: (time_step, dataset_name, time_value)
    """
    datasets = []
    #pattern = re.compile(rf'model {fracture_num} at time = (\d+)_([0-9\.e\+\-]+)')
    pattern = re.compile(rf"model {fracture_num} fracture at time = (\d+)_([0-9.eE+-]+)")

    for key in h5_file.keys():
        match = pattern.match(key)
        if match:
            time_step = int(match.group(1))
            time_value = float(match.group(2))
            datasets.append((time_step, key, time_value))

    # add new pattern for initial time to adjust crack data in metrics later..
    data_initial = []
    pattern = re.compile(rf"model {fracture_num} fracture initial")

    for key in h5_file.keys():
        match = pattern.match(key)
        if match:
            data_initial = h5_file[key][:]
    

    # Also check for final dataset (last key IS FINAL.. so ignore)
    # final_key = f'model {fracture_num} fracture final'
    # if final_key in h5_file.keys():
    #     # Assign a high time step number for final
    #     datasets.append((9999, final_key, float('inf')))
    
    # Sort by time step
    datasets.sort(key=lambda x: x[0])
    return datasets, data_initial

def process_fracture_groups(fracture_groups, output_dir, metrics_dir,
                            minValue=0.9, alpha=0.008,
                            plot_spider_plot=False):
    """
    Process fracture groups by computing crack metrics and optional plotting.

    Parameters
    ----------
    fracture_groups : dict
        Mapping from fracture_num -> list of (sample_num, filepath).
    output_dir : str
        Directory to save plots.
    metrics_dir : str
        Directory to save metric files (not currently used here).
    minValue : float
        Minimum value for plotting filter.
    alpha : float
        Transparency for plotting.
    plot_spider_plot : bool
        If True, generate spider plot visualizations.

    Returns
    -------
    all_metrics : dict
        Nested dict: all_metrics[fracture_num][sample_num] = list of metrics.
    """
    all_metrics = {}

    for fracture_num, sample_files in fracture_groups.items():
        print(f"Processing fracture {fracture_num} with {len(sample_files)} samples")
        all_metrics[fracture_num] = {}

        for sample_num, filepath in sample_files:
            print(f"  Processing sample {sample_num}")
            with h5py.File(filepath, 'r') as f:
                time_datasets, data_initial = extract_time_step_datasets(f, fracture_num)
                sample_metrics = []

                for time_step, dataset_name, time_value in time_datasets:
                    if dataset_name in f:
                        data = f[dataset_name][:]
                        metrics = calculate_crack_metrics(data, data_initial)
                        metrics['time_step'] = time_step
                        metrics['time_value'] = time_value
                        metrics['dataset_name'] = dataset_name
                        sample_metrics.append(metrics)
                    else:
                        print(f"    Warning: Dataset {dataset_name} not found")

                all_metrics[fracture_num][sample_num] = sample_metrics

        if plot_spider_plot:
            plt.figure(figsize=(8, 8))
            for sample_num, filepath in sample_files:
                try:
                    with h5py.File(filepath, 'r') as f:
                        dataset_name = f'model {fracture_num} fracture final'
                        if dataset_name in f:
                            data = f[dataset_name][:]
                            plot_damage_lines_within_range(
                                data, minValue=minValue, maxValue=2, alpha=alpha
                            )
                        else:
                            print(f"Missing dataset {dataset_name} in {filepath}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Fracture {fracture_num} Comparison Across Samples')
            plt.xlim(0, 0.25)
            plt.ylim(0, 0.25)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'fracture_{minValue}_{alpha}_{fracture_num}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()

    return all_metrics

def plot_fracture_metrics(all_metrics, metrics_dir, 
                          num_samples=125, 
                          cmap=plt.cm.turbo, 
                          filter_samples=None):
    """
    Generate summary plots of metrics evolution for each fracture.

    Parameters
    ----------
    all_metrics : dict
        Dictionary of the form {fracture_num: {sample_num: [metrics_dicts...]}}.
    metrics_dir : str
        Directory to save plots.
    num_samples : int, optional
        Total number of samples for colormap normalization. Default = 125.
    cmap : matplotlib colormap, optional
        Colormap used for plotting samples. Default = plt.cm.turbo.
    filter_samples : list[int], optional
        If provided, only these sample numbers will be plotted.
    """
    os.makedirs(metrics_dir, exist_ok=True)

    for fracture_num, fracture_data in all_metrics.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Fracture {fracture_num} - Metrics Evolution Over Time')

        for i, (sample_num, sample_metrics) in enumerate(fracture_data.items()):
            if filter_samples is not None and int(sample_num) not in filter_samples:
                continue    

            if len(sample_metrics) == 0:
                continue

            print(f'Working on fracture {fracture_num}, sample {sample_num}')

            # filter out inf values
            valid = [m for m in sample_metrics if m['time_value'] != float('inf')]

            time_values = [m['time_value'] for m in valid]
            total_cracks_ = [m['total_cracks'] for m in valid]
            largest_cluster_fraction_ = [m['largest_cluster_fraction'] for m in valid]
            max_distances_ = [m['max_euclidean_distance'] for m in valid]
            avg_orientation_ = [m['avg_orientation'] for m in valid]
            num_clusters_ = [m['num_clusters'] for m in valid]

            if len(time_values) > 0:
                color = cmap(i / num_samples)

                axes[0,0].plot(time_values, total_cracks_, '.-', alpha=0.7, 
                               label=f'Sample {sample_num}', color=color)
                axes[0,1].plot(time_values, largest_cluster_fraction_, alpha=0.7, 
                               label=f'Sample {sample_num}', color=color)
                axes[1,0].plot(time_values, max_distances_, alpha=0.7, 
                               label=f'Sample {sample_num}', color=color)
                axes[1,1].plot(time_values, num_clusters_, alpha=0.7, 
                               label=f'Sample {sample_num}', color=color)

                # axes[1,1].plot(time_values, avg_orientation_, alpha=0.7, 
                #                label=f'Sample {sample_num}', color=color)

        # Labels and titles
        axes[0,0].set_xlabel('Time'); axes[0,0].set_ylabel('Total Cracks')
        axes[0,0].set_title('Total Cracks vs Time'); axes[0,0].grid(True, alpha=0.3)

        axes[0,1].set_xlabel('Time'); axes[0,1].set_ylabel('Largest Cluster Fraction')
        axes[0,1].set_title('Largest Cluster vs Time'); axes[0,1].grid(True, alpha=0.3)

        axes[1,0].set_xlabel('Time'); axes[1,0].set_ylabel('Shortest-Longest Path')
        axes[1,0].set_title('Graph Diameter vs Time'); axes[1,0].grid(True, alpha=0.3)

        axes[1,1].set_xlabel('Time'); axes[1,1].set_ylabel('Number of Clusters')
        axes[1,1].set_title('Number of Clusters vs Time'); axes[1,1].grid(True, alpha=0.3)


        # axes[1,1].set_xlabel('Time'); axes[1,1].set_ylabel('Average Orientation')
        # axes[1,1].set_title('Average Orientation vs Time'); axes[1,1].grid(True, alpha=0.3)

        # One legend for all axes
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), borderaxespad=0)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # room for legend
        metrics_plot_path = os.path.join(metrics_dir, f'fracture_{fracture_num}_metrics_evolution.png')
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def summarize_all_metrics(all_metrics, early_fraction=0.2):
    """
    Summarize per (fracture, sample) into composite features.
    
    Returns: DataFrame with columns:
        fracture, sample, time_to_failure, total_cracks,
        early_clusters, longest_crack, crack_growth_rate
    """
    rows = []
    
    for fracture_num, sample_dict in all_metrics.items():
        for sample_num, timeline in sample_dict.items():
            if not timeline:
                continue
            
            final = timeline[-1]
            time_to_failure = final["time_value"]
            total_cracks = final["total_cracks"]
            longest_crack = final["max_euclidean_distance"]
            
            # Early cluster behavior
            cutoff = early_fraction * time_to_failure
            early = [entry["num_clusters"] for entry in timeline if entry["time_value"] <= cutoff]
            early_clusters = max(early) if early else timeline[0]["num_clusters"]
            
            # Crack growth rate
            times = [entry["time_value"] for entry in timeline]
            cracks = [entry["total_cracks"] for entry in timeline]
            
            if len(times) >= 2:
                # Simple slope (final - first) / (Δt)
                crack_growth_rate = (cracks[-1] - cracks[0]) / (times[-1] - times[0] + 1e-12)
            else:
                crack_growth_rate = 0.0
            
            rows.append({
                "fracture": fracture_num,
                "sample": sample_num,
                "time_to_failure": time_to_failure,
                "total_cracks": total_cracks,
                "early_clusters": early_clusters,
                "longest_crack": longest_crack,
                "crack_growth_rate": crack_growth_rate,
            })
    
    return pd.DataFrame(rows)

#========================================================================================
#========================================================================================
#========================================================================================
#========================================================================================
#========================================================================================
# MAIN
#========================================================================================
#========================================================================================
#========================================================================================
#========================================================================================
#========================================================================================

# Set up paths
base_dir = '../newTTF_localtesting/data/pp_results'
output_dir = '../newTTF_localtesting/results'
metrics_dir = '../newTTF_localtesting/results/metrics'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Regex to parse filenames
pattern = re.compile(r'frac_pull_z_Sample_(\d+)_(\d+)\.h5')

# Group files by fracture number
fracture_groups = defaultdict(list)
for fname in os.listdir(base_dir):
    match = pattern.match(fname)
    if match:
        sample_num, fracture_num = match.groups()
        filepath = os.path.join(base_dir, fname)
        fracture_groups[fracture_num].append((sample_num, filepath))

# Call the function
all_metrics = process_fracture_groups(
    fracture_groups,
    output_dir,
    metrics_dir,
    minValue=0.9,
    alpha=0.008,
    plot_spider_plot=False
)


# Save the metrics for each fracture
# with open("all_metrics.pkl", "wb") as f:
#     pickle.dump(all_metrics, f)


#

# Load
with open("all_metrics.pkl", "rb") as f:
    all_metrics_loaded = pickle.load(f)

# pick a continuous colormap (can try "viridis", "plasma", "coolwarm", etc.)
cmap = plt.cm.turbo  
num_samples = 125  # total samples

# lets get multiple fractures plotted to see p
filter_samples = None
filter_samples = [113]
plot_fracture_metrics(all_metrics, metrics_dir, num_samples=125, cmap=plt.cm.turbo, filter_samples=filter_samples)


df = summarize_all_metrics(all_metrics, early_fraction=0.2)

def compute_strength_scores(df, weights=None):
    """
    Compute normalized strength scores per (fracture, sample).
    
    Args:
        df : pd.DataFrame with columns
            ['fracture', 'sample', 'time_to_failure', 
             'total_cracks', 'early_clusters', 'longest_crack', 'crack_growth_rate']
        weights : dict, optional
            relative weight of each metric in the composite score.
            Example: {"time_to_failure": 1.0, "total_cracks": 1.0, ...}
            
    Returns:
        df_scores : DataFrame with normalized metrics + score per sample
        fracture_scores : DataFrame with average score per fracture
        overall_score : float (average across all fractures)
    """
    # Default weights (tweak as you like)
    if weights is None:
        weights = {
            "time_to_failure": 1.0,
            "total_cracks": 1.0,
            "early_clusters": 1.0,
            "longest_crack": 1.0,
            "crack_growth_rate": 1.0,
        }

    # Copy to avoid modifying original
    dfc = df.copy()
    
    # Normalize per fracture
    metrics = list(weights.keys())
    for metric in metrics:
        dfc[f"{metric}_norm"] = dfc.groupby("fracture")[metric].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )
    
    # Define direction of "goodness":
    # Higher time_to_failure = better → keep positive
    # Lower total_cracks, early_clusters, longest_crack, crack_growth_rate = better → invert
    invert = {"time_to_failure": False,
              "total_cracks": True,
              "early_clusters": True,
              "longest_crack": True,
              "crack_growth_rate": True}
    
    for metric in metrics:
        if invert[metric]:
            dfc[f"{metric}_norm"] = 1.0 - dfc[f"{metric}_norm"]
    
    # Weighted average score
    total_weight = sum(weights.values())
    dfc["score"] = sum(dfc[f"{m}_norm"] * w for m, w in weights.items()) / total_weight

    sample_scores = dfc.groupby("sample")["score"].mean()
    # # Average score per fracture
    # fracture_scores = dfc.groupby("fracture")["score"].mean().reset_index()
    
    # # Overall average
    # overall_score = fracture_scores["score"].mean()
    
    return dfc, sample_scores

dfc, sample_scores = compute_strength_scores(df, weights=None)

def plot_best_worst_samples(dfc, metrics_dir, sample_scores, fracture_groups, fracture_num, minValue=0.01, alpha=0.3):
    """
    Plot top 5 best and worst samples for a chosen fracture.
    
    Args:
        dfc : DataFrame returned from compute_strength_scores
        sample_scores : Series of average scores per sample
        fracture_groups : dict mapping fracture -> list of (sample, filepath)
        fracture_num : str or int, fracture to visualize
        minValue : float, min value for plotting
        alpha : float, transparency for plotting
    """
    # Ensure fracture_num is string to match fracture_groups keys
    fracture_num = str(fracture_num)
    if fracture_num not in fracture_groups:
        raise ValueError(f"Fracture {fracture_num} not found in fracture_groups")

    # Sort sample scores (global scores, averaged across fractures)
    ranked_samples = sample_scores.sort_values(ascending=False)

    top5 = ranked_samples.head(5).index.astype(str).tolist()
    #bottom5 = ranked_samples.tail(5).index.astype(str).tolist()
    bottom5 = ranked_samples.tail(5).sort_values().index.astype(str).tolist()
    
    # Extract available sample files for this fracture
    sample_files = {s: f for s, f in fracture_groups[fracture_num]}

    # Set up figure
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    fig.suptitle(f"Fracture {fracture_num}: Top 5 vs Worst 5 Samples", fontsize=16)

    # Helper function for plotting one panel
    def plot_sample(ax, sample_num, filepath, title):
        try:
            with h5py.File(filepath, "r") as f:
                dataset_name = f"model {fracture_num} fracture final"
                if dataset_name in f:
                    data = f[dataset_name][:]
                    plot_damage_lines_within_range(data, minValue=minValue, maxValue=2, alpha=alpha, ax=ax)
                    ax.set_title(title, fontsize=10)
                else:
                    ax.set_title(f"{title}\n(no dataset)", fontsize=10, color="red")
        except Exception as e:
            ax.set_title(f"{title}\n(error: {e})", fontsize=10, color="red")

    # Plot best 5
    for i, sample_num in enumerate(top5):
        filepath = sample_files.get(sample_num)
        title = f"Best {i+1}: Sample {sample_num} (score={sample_scores[sample_num]:.3f})"
        plot_sample(axes[i, 0], sample_num, filepath, title)

    # Plot worst 5
    for i, sample_num in enumerate(bottom5):
        filepath = sample_files.get(sample_num)
        title = f"Worst {i+1}: Sample {sample_num} (score={sample_scores[sample_num]:.3f})"
        plot_sample(axes[i, 1], sample_num, filepath, title)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    #plt.show()
    metrics_plot_path = os.path.join(metrics_dir, f'fracture_{fracture_num}_best_worst_samples.png')
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    #plt.savefig()

plot_best_worst_samples(dfc, metrics_dir, sample_scores, fracture_groups, fracture_num, minValue=0.9, alpha=1.0)

#potentially some other scoring metrics instead of weighted sum ..

# def percentile_norm(series):
#     """Return 0..1 percentile ranks (robust)."""
#     # using rank percentile:
#     return series.rank(pct=True).values

# def minmax_norm(series, clip=True):
#     mn, mx = series.min(), series.max()
#     if mx == mn:
#         out = np.zeros_like(series, dtype=float)
#     else:
#         out = (series - mn) / (mx - mn)
#     if clip:
#         out = np.clip(out, 0.0, 1.0)
#     return out


# def compute_composite_scores(df,
#                              method='weighted_sum',
#                              weights=None,
#                              norm='percentile'):
#     """
#     df: DataFrame with columns ['time_to_failure','total_cracks','early_clusters','longest_crack']
#     method: 'weighted_sum' or 'multiplicative' or 'pca'
#     norm: 'percentile' or 'minmax' or 'z'
#     returns: DataFrame with added 'score' column and component normalized columns.
#     """
#     dfc = df.copy().reset_index(drop=True)
#     # choose normalization function
#     if norm == 'percentile':
#         norm_fun = percentile_norm
#     elif norm == 'minmax':
#         norm_fun = minmax_norm
#     elif norm == 'z':
#         norm_fun = lambda s: (s - s.mean()) / (s.std() + 1e-12)
#     else:
#         raise ValueError("norm must be percentile/minmax/z")
    
#     # Normalize into 0..1 where higher=better
#     # time_to_failure: higher better
#     if norm == 'z':
#         # for z we will later translate to a score differently; but keep simple here
#         dfc['t_norm'] = norm_fun(dfc['time_to_failure'])
#         dfc['tc_norm'] = -norm_fun(dfc['total_cracks'])     # lower better -> invert
#         dfc['ec_norm'] = -norm_fun(dfc['early_clusters'])
#         dfc['l_norm'] = -norm_fun(dfc['longest_crack'])
#         # convert z -> percentile to keep on same scale
#         # fallback to percentile
#         dfc['t_norm'] = percentile_norm(dfc['time_to_failure'])
#         dfc['tc_norm'] = 1 - percentile_norm(dfc['total_cracks'])
#         dfc['ec_norm'] = 1 - percentile_norm(dfc['early_clusters'])
#         dfc['l_norm'] = 1 - percentile_norm(dfc['longest_crack'])
#     else:
#         dfc['t_norm'] = norm_fun(dfc['time_to_failure'])
#         dfc['tc_norm'] = 1.0 - norm_fun(dfc['total_cracks'])
#         dfc['ec_norm'] = 1.0 - norm_fun(dfc['early_clusters'])
#         dfc['l_norm'] = 1.0 - norm_fun(dfc['longest_crack'])
    
#     # default weights
#     if weights is None:
#         weights = {'time': 0.4, 'total_cracks': 0.2, 'early_clusters': 0.2, 'longest': 0.2}
#     # normalize weights
#     totw = sum(weights.values())
#     for k in weights:
#         weights[k] = weights[k] / totw
    
#     if method == 'weighted_sum':
#         dfc['score'] = (weights['time'] * dfc['t_norm'] +
#                         weights['total_cracks'] * dfc['tc_norm'] +
#                         weights['early_clusters'] * dfc['ec_norm'] +
#                         weights['longest'] * dfc['l_norm'])
#     elif method == 'multiplicative':
#         # small epsilon to avoid zeroing
#         eps = 1e-6
#         dfc['score'] = ((dfc['t_norm'] + eps) ** weights['time']) * \
#                        ((dfc['tc_norm'] + eps) ** weights['total_cracks']) * \
#                        ((dfc['ec_norm'] + eps) ** weights['early_clusters']) * \
#                        ((dfc['l_norm'] + eps) ** weights['longest'])
#         # optionally rescale to 0..1:
#         dfc['score'] = (dfc['score'] - dfc['score'].min()) / (dfc['score'].max() - dfc['score'].min() + 1e-12)
#     else:
#         raise ValueError("Unknown method")
    
#     return dfc
