import numpy as np
import pandas as pd
from particle_filter import PersonalizedParticleFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Court dimensions constants (in meters)
COURT_LENGTH = 18.0  # Standard volleyball court length
COURT_WIDTH = 9.0    # Standard volleyball court width
NET_HEIGHT = 2.43     # Official net height for men's volleyball
NET_WIDTH = 1.0       # Width of the net poles

def draw_volleyball_court(ax):
    """Draw a 3D volleyball court on the given matplotlib axis"""
    hl = COURT_LENGTH / 2  # half length
    hw = COURT_WIDTH / 2   # half width

    # Court boundary (white lines)
    court_lines = [
        [(-hl, -hw, 0), (hl, -hw, 0)],  # Back line
        [(-hl, hw, 0), (hl, hw, 0)],    # Front line
        [(-hl, -hw, 0), (-hl, hw, 0)],  # Left sideline
        [(hl, -hw, 0), (hl, hw, 0)]    # Right sideline
    ]
    
    for line in court_lines:
        ax.plot(*zip(*line), color='white', linewidth=1)

    # Center line (dashed white)
    ax.plot([-hl, hl], [0, 0], [0, 0], 'white', linestyle='--', linewidth=1)

    # Attack lines (3 meters from center)
    attack_line = 3.0  
    ax.plot([attack_line, attack_line], [-hw, hw], [0, 0], 'white', linewidth=1)
    ax.plot([-attack_line, -attack_line], [-hw, hw], [0, 0], 'white', linewidth=1)

    # Net (red)
    ax.plot([-NET_WIDTH/2, NET_WIDTH/2], [0, 0], [NET_HEIGHT, NET_HEIGHT], 
            'red', linewidth=2)
    
    # Net poles
    ax.plot([0, 0], [-hw, -hw], [0, NET_HEIGHT], 'red', linewidth=2)
    ax.plot([0, 0], [hw, hw], [0, NET_HEIGHT], 'red', linewidth=2)

    # Set court appearance
    ax.set_facecolor('darkgreen')
    ax.grid(False)

def main():
    # Load the data
    try:
        data = pd.read_csv("tracking/runs/detect/train4/coordinates/coords_3d_all.csv")
    except FileNotFoundError:
        print("Error: Input CSV file not found!")
        return
    
    # Initialize results list
    results = []
    
    # Get first valid detection
    first_valid_idx = data['X'].first_valid_index()
    if first_valid_idx is None:
        print("Error: No valid detections found in the data!")
        return
        
    initial_detection = data.iloc[first_valid_idx][['X', 'Y', 'Z']].values
    
    # Initialize particle filter
    pf = PersonalizedParticleFilter(
        initial_state=initial_detection,
        num_particles=2000,
        process_noise_std=[0.2, 0.1],
        measurement_noise_std=0.3,
        gravity=-9.81
    )
    
    previous_time = data.iloc[first_valid_idx]["timestamp_sec"]
    
    # Process each detection
    for i in range(first_valid_idx + 1, len(data)):
        try:
            current_time = data.iloc[i]["timestamp_sec"]
            dt = max(0.001, current_time - previous_time)  # Prevent dt=0
            previous_time = current_time

            # Prediction step
            pf.predict(dt=dt)
            
            # Update step
            detection = data.iloc[i][['X', 'Y', 'Z']].values if not np.isnan(data.iloc[i]["X"]) else None
            pf.update_weights(detection)
            pf.resample()

            # Get estimation
            estimated_state = pf.estimate()
            
            # Store results
            results.append({
                'timestamp_sec': current_time,
                'X_est': estimated_state[0],
                'Y_est': estimated_state[1],
                'Z_est': estimated_state[2],
                'X_det': data.iloc[i]["X"],
                'Y_det': data.iloc[i]["Y"],
                'Z_det': data.iloc[i]["Z"]
            })
        except Exception as e:
            print(f"Error processing frame {i}: {str(e)}")
            continue

    # Convert results to DataFrame
    estimated_state_df = pd.DataFrame(results)
    
    # Save to CSV
    try:
        estimated_state_df.to_csv("tracking/runs/detect/train4/coordinates/estimated_states.csv", index=False)
        print("Estimated states saved to estimated_states.csv")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

    # Plotting
    fig = plt.figure(figsize=(14, 10)) 
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw court
    draw_volleyball_court(ax)
    
    # Plot detections and estimates
    ax.scatter(estimated_state_df['X_det'], estimated_state_df['Y_det'], estimated_state_df['Z_det'], 
               c='blue', label='Detected Coordinates', alpha=0.6, s=20)
    ax.plot(estimated_state_df['X_est'], estimated_state_df['Y_est'], estimated_state_df['Z_est'], 
            c='red', label='Estimated Trajectory', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X-axis (Court Length)', fontsize=12)
    ax.set_ylabel('Y-axis (Court Width)', fontsize=12)
    ax.set_zlabel('Z-axis (Height)', fontsize=12)
    ax.set_title('3D Volleyball Tracking with Particle Filter', fontsize=14)
    
    ax.set_xlim(-COURT_LENGTH/2, COURT_LENGTH/2)
    ax.set_ylim(-COURT_WIDTH/2, COURT_WIDTH/2)
    ax.set_zlim(0, 5)  # Up to 5 meters height
    
    # Add legend and adjust layout
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()