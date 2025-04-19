import numpy as np
from scipy.stats import norm

class ParticleFilter:
    """
    Particle Filter for 3D tracking with parabolic motion model.
    
    Parameters:
        initial_state (array-like): [x, y, z] initial position
        num_particles (int): Number of particles
        process_noise_std (float): Process noise standard deviation
        measurement_noise_std (float): Measurement noise standard deviation
        initial_state_std (float): Initial spread standard deviation
    """
    def __init__(self, initial_state, num_particles=1000, 
                 process_noise_std=0.5, measurement_noise_std=1.0,
                 initial_state_std=1.0):
        self.n_particles = num_particles
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.last_timestamp = None
        
        # Initialize particles around first detection with random velocities
        pos_noise = initial_state_std * np.random.randn(self.n_particles, 3)
        vel_noise = 1.0 * np.random.randn(self.n_particles, 3)
        self.particles = np.hstack([
            initial_state + pos_noise,
            vel_noise
        ])
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def predict(self, dt):
        """
        Predict next state with parabolic motion model (with gravity)
        
        Args:
            dt: time difference since last update
        """
        # State transition matrix for constant acceleration model
        F = np.array([
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Apply state transition
        self.particles = self.particles @ F.T
        
        # Apply gravity in Z axis
        self.particles[:,4] += -0.5*9.81*dt**2  # position z
        self.particles[:,5] += -9.81*dt         # velocity z
        
        # Add process noise
        process_noise = self.process_noise_std * np.random.randn(*self.particles.shape)
        self.particles += process_noise
    
    def update_weights(self, detection):
        """
        Update particle weights based on measurement
        
        Args:
            detection: [x, y, z] measured position
        Returns:
            weights: updated particle weights
        """
        if detection is None:
            return self.weights
            
        # Calculate Euclidean distance between particles and detection
        pos_errors = np.linalg.norm(self.particles[:,:3] - detection, axis=1)
        
        # Update weights using Gaussian PDF
        self.weights = norm.pdf(pos_errors, 0, self.measurement_noise_std)
        self.weights += 1e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)  # Normalize
        
        return self.weights
    
    def resample(self, weights=None):
        """Systematic resampling of particles"""
        if weights is None:
            weights = self.weights
            
        # Systematic resampling
        indices = np.arange(self.n_particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Avoid round-off error
        uniforms = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        new_indices = np.searchsorted(cumulative_sum, uniforms)
        
        self.particles = self.particles[new_indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def estimate(self):
        """Weighted mean estimate of particle states"""
        return np.average(self.particles[:,:3], weights=self.weights, axis=0)