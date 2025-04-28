import numpy as np
import pandas as pd
from utils.particle_filter import ParticleFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d

from utils.parameters import COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, NET_WIDTH, RESULTS_DIR

TRACKING_DIR = "tracking/"
outlier_threshold = 4.0  # Threshold for outlier detection in meters

from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np

def lowpass_filter(data, cutoff=5.0, fs=100.0, order=2, padlen=150):
    """
    Apply a zero-phase lowpass Butterworth filter to the data for smoother results.
    
    Args:
        data: DataFrame with columns ['X_est', 'Y_est', 'Z_est']
        cutoff: cutoff frequency in Hz (default: 2.0)
        fs: sampling frequency in Hz (default: 100.0)
        order: order of the filter (default: 4, higher for steeper rolloff)
        padlen: padding length for filtfilt (default: 150, higher for smoother edges)
    
    Returns:
        DataFrame with filtered values (same columns as input)
    
    Note:
        - Uses forward-backward filtering (filtfilt) for zero phase delay
        - Higher order gives steeper rolloff but may introduce artifacts
        - Larger padlen reduces edge effects but increases computation
    """
    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    required_cols = ['X_est', 'Y_est', 'Z_est']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns {required_cols}")
    
    if len(data) < order * 3:
        raise ValueError(f"Need at least {order * 3} samples for order {order} filter")
    
    # Design Butterworth filter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter to each coordinate
    filtered_data = data.copy()
    for col in required_cols:
        # Handle NaN values by interpolation before filtering
        clean_series = data[col].interpolate().ffill().bfill()
        
        # Apply zero-phase filtering
        filtered_data[col] = filtfilt(
            b, a, 
            clean_series,
            padlen=min(padlen, len(data) - 1)  # Ensure padlen isn't too large
        )
    
    return filtered_data

def draw_volleyball_court(ax):
    """Draw a 3D volleyball court on the given matplotlib axis"""
    hl = COURT_LENGTH / 2  # half length
    hw = COURT_WIDTH / 2   # half width

    # Court boundary 
    court_lines = [
        [(-hl, -hw, 0), (hl, -hw, 0)],  # Back line
        [(-hl, hw, 0), (hl, hw, 0)],    # Front line
        [(-hl, -hw, 0), (-hl, hw, 0)],  # Left sideline
        [(hl, -hw, 0), (hl, hw, 0)]     # Right sideline
    ]
    
    for line in court_lines:
        ax.plot(*zip(*line), color='blue', linewidth=1)

    # Center line
    ax.plot([-hl, hl], [0, 0], [0, 0], 'blue', linestyle='--', linewidth=1)

    # Attack lines (3 meters from center)
    attack_line = 3.0  
    ax.plot([attack_line, attack_line], [-hw, hw], [0, 0], 'blue', linewidth=1)
    ax.plot([-attack_line, -attack_line], [-hw, hw], [0, 0], 'blue', linewidth=1)

    # Net
    ax.plot([-NET_WIDTH/2, NET_WIDTH/2], [0, 0], [NET_HEIGHT, NET_HEIGHT], 
            'blue', linewidth=2)
    
    # Net poles
    ax.plot([0, 0], [-hw, -hw], [0, NET_HEIGHT], 'blue', linewidth=2)
    ax.plot([0, 0], [hw, hw], [0, NET_HEIGHT], 'blue', linewidth=2)

def main():
    # Load the data
    try:
        data = pd.read_csv(RESULTS_DIR + "coords_3d_all.csv")
    except FileNotFoundError:
        print("Error: Input CSV file not found!")
        return
    
    # Preprocess: filter invalid points and sort by timestamp
    
    valid_detections = data[['timestamp_sec', 'X', 'Y', 'Z']].dropna()
    valid_detections = valid_detections.sort_values('timestamp_sec')
    
    # Initialize PF with first valid detection
    if len(valid_detections) == 0:
        print("Error: No valid detections found!")
        return
        
    first_detection = valid_detections.iloc[0]
    #print(f"First detection: {first_detection}")
    initial_state = first_detection[['X', 'Y', 'Z']].values
    #print(f"Initial state: {initial_state}")
    
    pf = ParticleFilter(
        initial_state=initial_state,
        num_particles=1000,
        process_noise_std=0.5,
        measurement_noise_std=1.0,
        initial_state_std=1.0
    )
    pf.last_timestamp = first_detection['timestamp_sec']
    
    results = []
    
    # Add first detection and estimate (they will be the same)
    results.append({
        'timestamp': first_detection['timestamp_sec'],
        'X_det': first_detection['X'],
        'Y_det': first_detection['Y'],
        'Z_det': first_detection['Z'],
        'X_est': initial_state[0],
        'Y_est': initial_state[1],
        'Z_est': initial_state[2]
    })
    
    # Process remaining detections
    for _, row in valid_detections.iloc[1:].iterrows():
        timestamp = row['timestamp_sec']
        detection = row[['X', 'Y', 'Z']].values
        
        # Calculate dt from last timestamp
        dt = timestamp - pf.last_timestamp
        pf.last_timestamp = timestamp
        
        # Predict next state
        pf.predict(dt)
        
        # Outlier check based on last estimate
        if results:  # If we already have at least one estimate
            last_estimate = np.array([results[-1]['X_est'], results[-1]['Y_est'], results[-1]['Z_est']])
            distance = np.linalg.norm(detection - last_estimate)
            
            if distance > outlier_threshold:
                # If it's an outlier, use only prediction
                estimated_state = pf.estimate()
                results.append({
                    'timestamp': timestamp,
                    'X_det': None,
                    'Y_det': None,
                    'Z_det': None,
                    'X_est': estimated_state[0],
                    'Y_est': estimated_state[1],
                    'Z_est': estimated_state[2]
                })
                continue
        
        # Update filter with detection
        weights = pf.update_weights(detection)
        pf.resample(weights)
        estimated_state = pf.estimate()
        
        results.append({
            'timestamp': timestamp,
            'X_det': detection[0],
            'Y_det': detection[1],
            'Z_det': detection[2],
            'X_est': estimated_state[0],
            'Y_est': estimated_state[1],
            'Z_est': estimated_state[2]
        })    
    
    # Convert to DataFrame
    estimated_state_df = pd.DataFrame(results)
    
    # Apply smoothing
    estimates_lpf = lowpass_filter(estimated_state_df)
    estimates_lpf = estimates_lpf[['timestamp', 'X_est', 'Y_est', 'Z_est']].copy()

    # Rename columns from *_est to *_smooth
    estimates_lpf = estimates_lpf.rename(columns={
        'X_est': 'X_smooth',
        'Y_est': 'Y_smooth',
        'Z_est': 'Z_smooth'
    })

    
    # Merge with original estimates: timestamp,X_det,Y_det,Z_det,X_est,Y_est,Z_est,X_smooth,Y_smooth,Z_smooth
    final_df = pd.merge(estimated_state_df, estimates_lpf, 
                         on='timestamp', how='left')
    
    # Visualization
    fig = plt.figure(figsize=(14, 10)) 
    ax = fig.add_subplot(111, projection='3d')
    draw_volleyball_court(ax)
    
    # Plot detections and estimates
    valid_detections = final_df.dropna(subset=['X_det'])
    ax.scatter(valid_detections['X_det'], valid_detections['Y_det'], valid_detections['Z_det'], 
               c='yellow', label='Detected Positions', alpha=0.6, s=20)
    ax.plot(final_df['X_est'], final_df['Y_est'], final_df['Z_est'], 
            c='orange', label='Estimated Trajectory', linewidth=2, alpha=0.5)
    ax.plot(final_df['X_smooth'], final_df['Y_smooth'], final_df['Z_smooth'], 
             c='red', label='Smoothed Trajectory', linewidth=3)
    
    # Set labels and title
    ax.set_xlabel('Court Length (X)', fontsize=12)
    ax.set_ylabel('Court Width (Y)', fontsize=12)
    ax.set_zlabel('Height (Z)', fontsize=12)
    ax.set_title('3D Volleyball Tracking with Particle Filter', fontsize=14)
    
    # Set court dimensions
    ax.set_xlim(-COURT_LENGTH/2, COURT_LENGTH/2)
    ax.set_ylim(-COURT_WIDTH/2, COURT_WIDTH/2)
    ax.set_zlim(0, 5)  # Up to 5 meters height
    
    # Add legend and adjust view
    ax.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Save results
    estimated_state_df.to_csv(RESULTS_DIR + 'estimated_trajectory.csv', index=False)
    final_df.to_csv(RESULTS_DIR + 'smoothed_trajectory.csv', index=False)

    # Save plot
    fig.savefig(RESULTS_DIR + 'trajectory_plot.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()