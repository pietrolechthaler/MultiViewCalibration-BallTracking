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

import numpy as np
from scipy.stats import norm

class ParticleFilter2:
    def __init__(self, n_particles=1000):
        self.n_particles = n_particles
        self.particles = None
        self.weights = np.ones(n_particles) / n_particles
        
    def initialize(self, initial_pos):
        # Inizializza particelle attorno alla prima misura con velocità casuali
        pos_noise = 0.1 * np.random.randn(self.n_particles, 3)
        vel_noise = 1.0 * np.random.randn(self.n_particles, 3)
        self.particles = np.hstack([
            initial_pos + pos_noise,
            vel_noise
        ])
    
    def predict(self, dt):
        # Dinamica del sistema (moto parabolico con gravità)
        F = np.array([
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Applica gravità in Z
        self.particles = self.particles @ F.T
        self.particles[:,4] += -0.5*9.81*dt**2  # pos z
        self.particles[:,5] += -9.81*dt         # vel z
        
        # Aggiungi rumore di processo
        process_noise = 0.1 * np.random.randn(*self.particles.shape)
        self.particles += process_noise
    
    def update(self, observation):
        # Calcola likelihood (distanza euclidea)
        obs_pos = observation[:3]
        pos_errors = np.linalg.norm(self.particles[:,::2] - obs_pos, axis=1)
        
        # Update weights (uso una gaussiana per la likelihood)
        self.weights = norm.pdf(pos_errors, 0, 0.5)
        self.weights += 1e-300  # Evita pesi nulli
        self.weights /= np.sum(self.weights)
        
        # Resampling (systematic resampling)
        indices = np.arange(self.n_particles)
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Avoid round-off error
        uniforms = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        new_indices = np.searchsorted(cumulative_sum, uniforms)
        self.particles = self.particles[new_indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def estimate(self):
        # Media pesata delle particelle
        return np.average(self.particles, weights=self.weights, axis=0)
    


class ParticleFilter:
    """
    Particle Filter per tracking 3D con timestamp
    
    Parameters:
        initial_state (array-like): stato iniziale [x, y, z]
        num_particles (int): numero di particelle
        process_noise_std (float): deviazione standard rumore di processo
        measurement_noise_std (float): deviazione standard rumore di misura
        initial_state_std (array-like): deviazione standard iniziale [x, y, z, vx, vy, vz]
    """
    def __init__(self, initial_state: np.ndarray, num_particles: int, 
                 process_noise_std: float, measurement_noise_std: float,
                 initial_state_std: np.ndarray):
        self.num_particles = num_particles
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        
        # Inizializza particelle con stato [x,y,z,vx,vy,vz]
        self.particles = np.random.normal(
            np.concatenate([initial_state, [0, 0, 0]]),  # Parte da velocità zero
            initial_state_std,
            size=(num_particles, 6)
        )
        self.last_timestamp = None

    def predict(self, dt: float):
        """Predici lo stato successivo dato l'intervallo di tempo dt"""
        noise = np.random.normal(0, self.process_noise_std, size=self.particles.shape)
        
        # Aggiorna posizione basata su velocità
        self.particles[:, :3] += self.particles[:, 3:] * dt + noise[:, :3]
        # Aggiorna velocità con rumore
        self.particles[:, 3:] += noise[:, 3:]

    def update_weights(self, detection: np.ndarray) -> np.ndarray:
        """Aggiorna i pesi basati sulla detection"""
        distances = np.linalg.norm(self.particles[:, :3] - detection, axis=1)
        weights = np.exp(-0.5 * (distances / self.measurement_noise_std)**2)
        weights_sum = weights.sum()
        
        if weights_sum > 0:
            weights /= weights_sum
        else:
            weights = np.ones(len(self.particles)) / len(self.particles)
        return weights

    def resample(self, weights: np.ndarray):
        """Ricampiona le particelle basandosi sui pesi"""
        indices = np.random.choice(len(self.particles), size=len(self.particles), p=weights)
        self.particles = self.particles[indices]

    def estimate(self):
        """Stima lo stato corrente"""
        return self.particles[:, :3].mean(axis=0)

import numpy as np

class ParticleFilterAdvanced:
    """
    3D Particle Filter with velocity, adaptive noise, gravity handling, and timestamp management.
    
    Parameters:
        initial_state (array-like): [x, y, z] initial position.
        initial_timestamp (float): Initial timestamp for tracking.
        num_particles (int): Number of particles.
        process_noise_std (array-like): [pos_noise_std, vel_noise_std] for position and velocity.
        measurement_noise_std (float): Noise standard deviation for detections.
        initial_spread_std (array-like): [pos_spread, vel_spread] for initial particle distribution.
        gravity (float): Gravitational acceleration (default: -9.81 m/s² on z-axis).
    """
    def __init__(self, initial_state, initial_timestamp, num_particles=1000, 
                 process_noise_std=[0.1, 0.05], measurement_noise_std=0.5, 
                 initial_spread_std=[0.5, 0.1], gravity=-9.81):
        self.num_particles = num_particles
        self.gravity = np.array([0, 0, gravity])  # Gravity vector (z-axis)
        self.process_noise_std = np.concatenate([
            np.full(3, process_noise_std[0]),  # Position noise
            np.full(3, process_noise_std[1])   # Velocity noise
        ])
        self.measurement_noise_std = measurement_noise_std
        self.last_timestamp = initial_timestamp

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

    def predict(self, current_timestamp, acceleration=np.zeros(3)):
        """
        Predict next state using physics model (velocity + acceleration).
        
        Parameters:
            current_timestamp (float): Current timestamp for delta time calculation.
            acceleration (array-like): External acceleration (e.g., ball hit).
        """
        # Calculate time difference since last update
        dt = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp

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
        return {
            'position': np.average(self.particles[:, :3], weights=self.weights, axis=0),
            'velocity': np.average(self.particles[:, 3:], weights=self.weights, axis=0),
            'timestamp': self.last_timestamp
        }