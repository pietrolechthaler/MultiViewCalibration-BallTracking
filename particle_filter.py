import numpy as np

class PersonalizedParticleFilter:
    """
    3D Particle Filter with velocity, adaptive noise, and gravity handling.
    
    Parameters:
        initial_state (array-like): [x, y, z] initial position.
        initial_velocity (array-like): [vx, vy, vz] initial velocity (optional).
        num_particles (int): Number of particles.
        process_noise_std (array-like): [pos_noise_std, vel_noise_std] for position and velocity.
        measurement_noise_std (float): Noise standard deviation for detections.
        initial_spread_std (array-like): [pos_spread, vel_spread] for initial particle distribution.
        gravity (float): Gravitational acceleration (default: -9.81 m/s² on z-axis).
    """
    def __init__(self, initial_state, num_particles=1000, 
                 process_noise_std=[0.1, 0.05], measurement_noise_std=0.5, 
                 initial_spread_std=[0.5, 0.1], gravity=-9.81):
        self.num_particles = num_particles
        self.gravity = np.array([0, 0, gravity])  # Gravity vector (z-axis)
        self.process_noise_std = np.concatenate([
            np.full(3, process_noise_std[0]),  # Position noise
            np.full(3, process_noise_std[1])   # Velocity noise
        ])
        self.measurement_noise_std = measurement_noise_std

        # Initialize particles: [x, y, z, vx, vy, vz] with spread
        self.particles = np.random.normal(
            loc=np.concatenate([initial_state, [0, 0, 0]]),  # Default zero velocity
            scale=np.concatenate([
                np.full(3, initial_spread_std[0]),  # Position spread
                np.full(3, initial_spread_std[1])   # Velocity spread
            ]),
            size=(num_particles, 6)
        )
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dt=1.0, acceleration=np.zeros(3)):
        """
        Predict next state using physics model (velocity + acceleration).
        
        Parameters:
            dt (float): Time step.
            acceleration (array-like): External acceleration (e.g., ball hit).
        """
        # Apply gravity + external acceleration
        total_acceleration = self.gravity + acceleration

        # Update velocity and position with noise
        noise = np.random.normal(0, self.process_noise_std, size=self.particles.shape)
        
        self.particles[:, 3:] += total_acceleration * dt + noise[:, 3:]  # Velocity
        self.particles[:, :3] += self.particles[:, 3:] * dt + noise[:, :3]  # Position

    def update_weights(self, detection):
        """
        Update weights based on detection likelihood (Mahalanobis distance for robustness).
        """
        if detection is not None:
            # Mahalanobis distance (considers measurement noise)
            diff = self.particles[:, :3] - detection
            squared_dist = np.sum((diff / self.measurement_noise_std) ** 2, axis=1)
            self.weights = np.exp(-0.5 * squared_dist)
            
            # Handle zero weights (avoid NaN)
            if np.sum(self.weights) > 0:
                self.weights /= np.sum(self.weights)
            else:
                self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            # No detection: maintain current weights but add entropy
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        """Systematic resampling with noise injection to avoid particle impoverishment."""
        indices = np.random.choice(
            self.num_particles, 
            size=self.num_particles, 
            p=self.weights
        )
        self.particles = self.particles[indices]
        
        # Add small noise to diversify particles
        self.particles += np.random.normal(0, 0.1 * self.process_noise_std, size=self.particles.shape)
        
        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """Weighted mean of particles (more accurate than simple mean)."""
        return np.average(self.particles[:, :3], weights=self.weights, axis=0)
