import numpy as np
import freud
import matplotlib
import gsd.hoomd
import re, os
from scipy.optimize import curve_fit
import quaternion
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

def plot_quantities(traj, draw='all', period=1, plot_smoothed=True, label=None, color='k', linestyle='-', scale=1, alpha=1, real_start=True, window_size=10, axs=None, additional_plot_params=None):
    # Create figure and axes if not provided, ensuring flexibility for standalone usage
    if axs is None:
        fig, axs = plt.subplots(len(draw) if draw != 'all' else 1)
        axs = np.array(axs).flatten()  # Ensure axs is an array for consistency

    quantities = {}  # Dictionary to store the quantities to be plotted
    timestep = []  # List to store the timesteps
    print_quantities = True  # Flag to control printing of quantities information

    for frame in traj[::period]:
        # Determine the current timestep based on whether to use real start or relative to the first frame
        current_step = frame.configuration.step if real_start else frame.configuration.step - traj[0].configuration.step
        timestep.append(current_step)

        # Iterate through all logged quantities in the current frame
        for key, value in frame.log.items():
            quantity_name = key.split('/')[-1]
            # Print quantities information only once
            if quantity_name in ['degrees_of_freedom', 'translational_degrees_of_freedom', 'rotational_degrees_of_freedom', 'num_particles'] and print_quantities:
                print(f"{quantity_name} = {value[0]}")
            # Skip 'timestep' and process other quantities based on 'draw' parameter
            elif quantity_name != 'timestep' and (draw == 'all' or quantity_name in draw):
                quantities.setdefault(quantity_name, []).append(value[0] * scale if quantity_name == 'potential_energy' else value[0])

        print_quantities = False  # Ensure quantities are printed only for the first frame

    # Determine which quantities to plot based on the 'draw' parameter
    quant_to_draw = quantities.keys() if draw == 'all' else draw

    # Check if the number of axes matches the number of quantities to draw
    assert len(axs) >= len(quant_to_draw), f'Number of axes ({len(axs)}) does not match the number of quantities to draw ({len(quant_to_draw)})'
    
    # Prepare additional plotting parameters if provided
    plot_params = additional_plot_params if additional_plot_params else {}
    # Plot each quantity in the determined set of quantities
    for i, quantity in enumerate(quant_to_draw):
        axs[i].plot(timestep, quantities[quantity], label=label, color=color, linestyle=linestyle, alpha=alpha, **plot_params)
        
        # Apply smoothing if requested and the data series is long enough
        if plot_smoothed and len(quantities[quantity]) > window_size:
            # Calculate the smoothed series
            smoothed = np.convolve(quantities[quantity], np.ones(window_size)/window_size, mode='valid')
            # Adjust timestep for smoothed series to match its length
            smoothed_timestep = timestep[int(window_size/2):-int(window_size/2)+1] if len(timestep) == len(smoothed) else timestep
            axs[i].plot(smoothed_timestep, smoothed, c='grey', **plot_params)

    return timestep, quantities

def get_cores_coords(frame):
    #return coordinates of the cores from the frame
    points = frame.particles.position
    cores = points[frame.particles.typeid==0]
    return cores

def get_cores_images(frame):
    #return images of the cores (needed for periodic boundary conditions)
    image = frame.particles.image
    image_core = image[frame.particles.typeid==0]
    return image_core

def get_cores_coords(frame):
    # Extracts the core particle positions from a frame
    points = frame.particles.position
    cores = points[frame.particles.typeid == 0]
    return cores

def get_Qls(traj, ls=[i for i in range(12)], frame=-1):
    """
    Calculate the Steinhardt order parameters Ql for particles in a specified frame of a trajectory.
    
    Parameters:
    - traj: The trajectory data loaded from a GSD file or similar.
    - ls: A list of integers specifying the order l of the Ql parameters to calculate.
    - frame: The index of the frame in the trajectory from which to calculate Ql. Defaults to the last frame.
    
    Returns:
    - A NumPy array containing the calculated Ql values for each specified l, for particles in the frame.
    """
    # Ensure the frame index is within the trajectory bounds
    if not -len(traj) <= frame < len(traj):
        raise IndexError("Frame index out of bounds.")
    
    # Get core particle positions and the simulation box from the specified frame
    cores = get_cores_coords(traj[frame])
    box = traj[frame].configuration.box
    
    # Compute the Voronoi diagram for the particle positions
    voro = freud.locality.Voronoi()
    voro.compute((box, cores))
    
    Qls = []
    for l in ls:
        # Initialize the Steinhardt order parameter calculation for order l
        ql = freud.order.Steinhardt(l, average=True, weighted=True)
        # Compute Ql values using the neighbors list from the Voronoi computation
        Qls.append(ql.compute((box, cores), neighbors=voro.nlist).particle_order)
    
    return np.array(Qls)

def PCA(m, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the dataset `m`.

    Parameters:
    - m: A 2D NumPy array of shape (n_features, n_samples), where n_features is the number of features,
         and n_samples is the number of samples.
    - n_components: The number of principal components to return.

    Returns:
    - m_reduced: The dataset projected onto the first `n_components` principal components.
    """

    # Calculate the covariance matrix of the dataset
    cov_matrix = np.cov(m, rowvar=False)

    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the subset of eigenvectors that corresponds to the top `n_components` principal components
    eigenvector_subset = sorted_eigenvectors[:, :n_components]

    # Project the data onto the selected principal components
    m_reduced = np.dot(m - np.mean(m, axis=0), eigenvector_subset)

    return m_reduced

def get_msd(traj):
    """
    Calculate the Mean Squared Displacement (MSD) for a given trajectory.
    
    Parameters:
    - traj: A trajectory loaded from a simulation output file (e.g., GSD file with HOOMD-blue).
    
    Returns:
    - A NumPy array containing the MSD calculated for each timestep in the trajectory.
    """
    # Preallocate arrays to store positions and images for all frames
    data_msd = np.empty((len(traj), len(get_cores_coords(traj[0])), 3), dtype=np.float32)
    images_msd = np.empty_like(data_msd, dtype=np.int32)
    
    # Populate the preallocated arrays with positions and image flags
    for i, frame in enumerate(traj):
        data_msd[i] = get_cores_coords(frame)
        images_msd[i] = get_cores_images(frame)
    
    # Extract the box dimensions from the first frame of the trajectory
    box = traj[0].configuration.box
    
    # Use freud's MSD module to compute the mean squared displacement
    msd_calculator = freud.msd.MSD(box=box, mode='window')
    msd = msd_calculator.compute(data_msd, images=images_msd).msd
    
    return msd

class Bonds:

    """
    The Bonds class is designed to analyze and manage bond information for a single frame of a particle simulation. It allows for the identification and characterization of bonds between core particles and associated patches within the specified simulation frame. Key functionalities include filtering particles by region, calculating bond pairs, and counting neighbor bonds, catering to both rigid and flexible bonding scenarios.

    Attributes:
    - frame: The simulation frame from which particle and bond information is extracted.
    - box: The dimensions of the simulation box, extracted from the frame.
    - r_bond: The cutoff radius for considering two particles as bonded.
    - region: An optional specification of a subregion within the simulation box for focused analysis.
    - cores_coords: Coordinates of core particles after applying any specified region filter.
    - patches_coords: Coordinates of patch particles, similarly filtered by region if specified.
    - patches_body: An array identifying the body each patch belongs to, used in rigid simulations.
    - n_cores: The number of core particles identified in the frame.
    - n_patches: The number of patch particles identified.
    - bonds: The calculated bonds based on the cutoff radius and particle positions.

    The class supports analyzing systems with or without rigid body dynamics and allows for the analysis within specific regions of interest within the simulation box.
    """

    def __init__(self, frame, r_bond=0.24, rigid=False, region=None):
        self.frame = frame
        self.box = frame.configuration.box
        self.r_bond = r_bond
        self.region = region 

        coords = self.frame.particles.position
        typeid = self.frame.particles.typeid
        self.cores_coords = coords[typeid==0]
        self.patches_coords = coords[typeid==1]

        if rigid:
            self.patches_body = self.frame.particles.body[self.frame.particles.typeid==1]
        else:
            self.patches_body = np.array([int(i//3) for i in range(len(self.patches_coords))])

        if self.region:
            mask = self._filter_region(self.cores_coords)
            true_indices = [i for i, value in enumerate(mask) if value]
            self.cores_coords = self.cores_coords[mask]
            self.patches_coords = self.patches_coords[np.isin(self.patches_body, true_indices)]

        self.n_cores = len(self.cores_coords)
        self.n_patches = len(self.patches_coords)
        self.bonds = self._get_bonds()

    def _filter_region(self, coords):
        # Implement the logic to filter out particles outside the region
        # For example, if region = [x_min, x_max, y_min, y_max, z_min, z_max]
        mask = (coords[:, 0] >= self.region[0]) & (coords[:, 0] <= self.region[1]) & (coords[:, 1] >= self.region[2]) & (coords[:, 1] <= self.region[3]) & (coords[:, 2] >= self.region[4]) & (coords[:, 2] <= self.region[5])
        return mask

    def _get_bonds(self):
        """
        Calculate bonds between particles based on a specified cutoff radius (r_bond).
    
        This method utilizes freud's AABBQuery for efficient spatial querying to find
        pairs of particles (patches) that are within the cutoff radius of each other,
        excluding self-interactions. The resulting bonds are determined based on 
        spatial proximity without considering other potential criteria for bonding.
    
        Returns:
        - A freud.locality.NeighborList object representing the bonds between particles.
        """
        # Initialize AABBQuery with the simulation box and patch coordinates
        # for efficient spatial queries.
        aq = freud.locality.AABBQuery(self.box, self.patches_coords)

        # Perform the query to find neighbors within r_bond distance, excluding
        # self-interactions (i.e., a particle cannot bond with itself).
        query_result = aq.query(self.patches_coords, {'r_max': self.r_bond, 'exclude_ii': True})

        # Convert the query results into a NeighborList, which efficiently
        # represents pairs of bonded particles.
        bonds = query_result.toNeighborList()
        return bonds

    def get_patches_pairs(self):
        
        """
        Identifies unique pairs of bonded patches.

        Returns:
        - A numpy array of unique, sorted patch-patch pair indices.
        """
        # Ensure the bonds are sorted to avoid duplicates like (1,2) and (2,1)
        sorted_bonds = np.sort(self.bonds[:], axis=1)
        # Use np.unique to remove duplicate pairs
        unique_pairs = np.unique(sorted_bonds, axis=0)
        return unique_pairs

    def get_cores_pairs(self):
        """
        Identifies unique pairs of cores that correspond to the bonded patches.

        Returns:
        - A numpy array of unique, sorted core-core pair indices.
        """
        patch_pairs = self.get_patches_pairs()
        # Map patch pairs back to their corresponding core pairs
        core_pairs = np.sort(self.patches_body[patch_pairs], axis=1)
        unique_core_pairs = np.unique(core_pairs, axis=0)
        return unique_core_pairs

    def _count_neighbors(self, pairs, n):
        """
        Counts the number of neighbors for each particle based on given pairs.

        Parameters:
        - pairs: An array of particle pairs.
        - n: Total number of particles.

        Returns:
        - A numpy array with the count of neighbors for each particle.
        """
        # Flatten the pairs array to count occurrences of each particle
        all_indices = pairs.flatten()
        counts = np.bincount(all_indices, minlength=n)
        return counts

    def count_cores_neighbors(self):
        """
        Counts the neighbors for each core particle, excluding intra-core bonds.

        Returns:
        - A numpy array with the count of neighboring cores for each core particle.
        """
        core_pairs = self.get_cores_pairs()
        # Filter out intra-core pairs for counting inter-core neighbors
        inter_core_pairs = core_pairs[core_pairs[:, 0] != core_pairs[:, 1]]
        return self._count_neighbors(inter_core_pairs.flatten(), self.n_cores)

    def count_intra_cores(self):
        """
        Counts intra-core bonds where patches from the same core form bonds.

        Returns:
        - A numpy array with counts of intra-core bonded patches for each core.
        """
        core_pairs = self.get_cores_pairs()
        # Keep only intra-core pairs
        intra_core_pairs = core_pairs[core_pairs[:, 0] == core_pairs[:, 1]]
        return self._count_neighbors(intra_core_pairs[:, 0], self.n_cores)

    def count_patches_neighbors(self):
        """
        Counts the neighbors for each patch based on direct patch-patch bonds.

        Returns:
        - A numpy array with the count of neighboring patches for each patch.
        """
        patch_pairs = self.get_patches_pairs()
        return self._count_neighbors(patch_pairs.flatten(), self.n_patches)
        
class Bonds_in_time:

    """
    The Bonds_in_time class extends bond analysis across multiple frames of a particle simulation, enabling the study of bond dynamics over time. It utilizes the Bonds class to analyze each frame individually and aggregates this information to provide insights into how bonds form, persist, and break throughout the trajectory of the simulation. This class is particularly useful for understanding the temporal stability of structures and the kinetics of bonding events in dynamic systems.

    Attributes:
    - traj: The entire trajectory of the simulation, consisting of multiple frames.
    - length: The number of frames in the trajectory.
    - times: A list of timestep indices for each frame in the trajectory, facilitating temporal analysis.
    - bonds_list: A list of Bonds objects, each corresponding to a frame in the trajectory, used for detailed frame-by-frame bond analysis.

    Methods include capabilities to count the number of neighbors for cores and patches over time, analyze the distribution of neighbor counts, and calculate bond lifetimes, providing a comprehensive view of bonding dynamics in particle simulations.
    """

    def __init__(self, traj, r_bond=0.24, rigid=False, region=None):
        self.traj = traj
        self.length = len(traj)
        self.times = [frame.configuration.step for frame in traj]
        self.bonds_list = [Bonds(frame, r_bond, rigid=rigid, region=region) for frame in traj]

    def _count_neighbors_generic(self, cores_or_patches='cores'):
        """
        Count the neighbors for each particle, dynamically determining the maximum number of neighbors.
        
        Args:
        - cores_or_patches (str): Specifies the type of particles to count neighbors for ('cores', 'intra_cores', or 'patches').
        
        Returns:
        - A numpy array where each row represents a specific number of neighbors, and each column a frame in the trajectory.
        """
        # Initialize a list to collect all neighbor counts
        all_neighbor_counts = []

        # Fetch the appropriate method based on cores_or_patches
        method_map = {
            'cores': lambda bond: bond.count_cores_neighbors(),
            'intra_cores': lambda bond: bond.count_intra_cores(),
            'patches': lambda bond: bond.count_patches_neighbors()
        }

        if cores_or_patches not in method_map:
            raise ValueError(f"Invalid particle type: {cores_or_patches}")

        getter_method = method_map[cores_or_patches]

        # Iterate through each bond object and collect neighbor counts
        for bond in self.bonds_list:
            all_neighbor_counts.append(getter_method(bond))

        # Flatten the list and find the max number of neighbors to define the array size
        all_counts_flattened = np.concatenate(all_neighbor_counts)
        max_neighbors = np.max(all_counts_flattened) + 1 if len(all_counts_flattened) > 0 else 0

        # Initialize the array to hold neighbor counts with dynamic size
        num_neighbors = np.zeros((max_neighbors, self.length), dtype=int)

        # Populate the array
        for i, neighbor_counts in enumerate(all_neighbor_counts):
            unique, counts = np.unique(neighbor_counts, return_counts=True)
            num_neighbors[unique, i] = counts

        return num_neighbors
    
    def count_cores_neighbors(self):
            return self._count_neighbors_generic(cores_or_patches='cores')

    def count_intra_cores(self):
            return self._count_neighbors_generic(cores_or_patches='intra_cores')

    def count_patches_neighbors(self):
        return self._count_neighbors_generic(cores_or_patches='patches') 

    def calculate_bond_lifetimes(self):
        """
        Calculate the lifetimes of bonds throughout the trajectory.

        This method tracks the formation, persistence, and breaking of bonds across frames,
        providing insights into bond stability and dynamics over time.

        Returns:
        - bond_presence_intervals: A dictionary where keys are tuples representing bonds (i, j),
        and values are lists of tuples, each indicating the presence interval (start_time, end_time)
        of the corresponding bond.
        """
        bond_presence_intervals = {}
        bond_temporary_presence = {}

        # Iterate over each frame to update bond presence intervals
        for i, frame in enumerate(self.bonds_list):
            current_time = self.times[i]
            # Extract current bonds as tuples and sort them to ensure consistency
            current_bonds = set(map(tuple, np.sort(frame.get_patches_pairs(), axis=1)))

            # Identify new and broken bonds
            new_bonds = current_bonds - bond_temporary_presence.keys()
            ended_bonds = set(bond_temporary_presence) - current_bonds

            # Update bond_presence_intervals for bonds that have ended in this frame
            for bond in ended_bonds:
                start_time = bond_temporary_presence.pop(bond)
                bond_presence_intervals.setdefault(bond, []).append((start_time, current_time))

            # Record start time for new bonds
            for bond in new_bonds:
                bond_temporary_presence[bond] = current_time

        # Handle bonds still active in the last frame by extending their presence to the last time
        last_time = self.times[-1]
        for bond, start_time in bond_temporary_presence.items():
            bond_presence_intervals.setdefault(bond, []).append((start_time, last_time))

        return bond_presence_intervals

class Packing_Fraction:

    """
    The Packing_Fraction class calculates the packing fraction profile across a specified simulation frames from a HOOMD-blue GSD file 

    Constructor Parameters:
    - file: Path to the GSD file containing the simulation trajectory.
    - n_patches: Number of patches per particle, used to calculate the total number of core particles.
    - frame: Specific frame to analyze (default is the last frame).
    - r_core: Radius of core particles, necessary for volume calculations.
    - window_size: Size of the smoothing window for the packing fraction profile.
    - N_bins: Number of bins to use for spatial division of the simulation box in the profile calculation.
    - average_over_frames: Number of frames over which to average the packing fraction profile (before reference frame).
    - phase_range: width of spatial range for identifying condensed and dilute phases.

    Methods:
    - calculate_phi: Returns the overall packing fraction of the system.
    - _get_phi_profile: Computes the spatial packing fraction profile and applies smoothing.
    - _centralize: Adjusts the profile to center the condensed phase 
    - draw_phi_profile: Visualizes the smoothed packing fraction profile with matplotlib.
    - get_phi_condensed: Calculates the average packing fraction in the condensed phase.
    - get_phi_dilute: Calculates the average packing fraction in the dilute phase.
    """

    def __init__(self, file, n_patches, frame=-1, r_core=0.5, window_size=5, N_bins=250, average_over_frames=10, phase_range=10):

        self.frame = frame
        self.traj = gsd.hoomd.open(file, 'rb')
        self.snapshot = self.traj[frame]
        self.Lx, self.Ly, self.Lz = self.snapshot.configuration.box[:3]

        self.r_core = r_core
        self.v_particle = (4/3) * np.pi * self.r_core**3 
        self.N = (self.snapshot.log['md/compute/ThermodynamicQuantities/num_particles']/(n_patches+1)) * n_patches

        self.window_size = window_size
        self.N_bins = N_bins
        self.average_over_frames=average_over_frames

        self.phase_range = phase_range
        self.phase_bins = int((N_bins/self.Lx)*self.phase_range)

        self._get_phi_profile()
        self._centralise()

    def calculate_phi(self, n_patches):
        V_box = self.Lx * self.Ly * self.Lz
        phi = self.v_particle*self.N/V_box
        return phi[0]
    
    def _get_phi_profile(self):   
        
        v_bin = (self.Lx/self.N_bins)*self.Ly*self.Lz

        #calculating profile with averaging over last frames
        phi_profiles = []
        for i in range(0, self.average_over_frames):
            coords_cores = get_cores_coords(self.traj[self.frame-i])
            xs = list(coords_cores[:,0])
            hist, bin_edges = np.histogram(xs, self.N_bins, range=(-self.Lx/2, self.Lx/2))
            phi_profiles.append(hist*self.v_particle/v_bin)
        phi_profiles = np.array(phi_profiles)
        phi_profile_mean = phi_profiles.mean(axis=0)
        self.profile = phi_profile_mean
    
        #smoothing profile
        window_size_bins = int((self.N_bins/self.Lx)*self.window_size)
        shift = (window_size_bins)//2
        if window_size_bins % 2 == 1:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
        else:
            bin_centers = bin_edges
        smoothed_phi_profile = np.convolve(np.concatenate([phi_profile_mean[-shift:],phi_profile_mean,phi_profile_mean[:shift]]), np.ones(window_size_bins)/window_size_bins, mode='valid')
        self.bin_centers = bin_centers
        self.smoothed_profile = smoothed_phi_profile

    def _centralise(self):
        gradient = (np.roll(self.smoothed_profile, -1) - np.roll(self.smoothed_profile, 1))/2
        max_grad_pos, min_grad_pos = np.argmax(gradient),np.argmin(gradient)
        center = int((max_grad_pos + min_grad_pos)/2)
        self.profile = np.roll(self.profile, int(self.N_bins/2-center))
        self.smoothed_profile = np.roll(self.smoothed_profile, int(self.N_bins/2-center))

    def draw_phi_profile(self, ax=None, color='b', alpha=0.1):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(self.bin_centers, self.smoothed_profile, color=color)
        ax.axvspan(-self.phase_range, self.phase_range, color='grey', alpha=alpha)
        ax.axvspan(-self.Lx/2, -self.Lx/2+self.phase_range, color='blue', alpha=alpha)
        ax.axvspan(self.Lx/2-self.phase_range, self.Lx/2, color='blue', alpha=alpha)

    def get_phi_condensed(self):
        return np.mean(self.profile[self.N_bins//2-self.phase_bins:self.N_bins//2+self.phase_bins])
    
    def get_phi_dilute(self):
        return np.mean(np.concatenate([self.profile[-self.phase_bins:], self.profile[:self.phase_bins]]))
            

class Phase_Diagram:

    """
    Analyzes phase behavior from simulation data, calculating the packing fraction profiles
    and identifying critical points of phase transition based on temperature variations.
    
    Attributes:
        directory (str): Directory containing GSD files for analysis.
        n_patches (int): Number of patches per particle.
        frame (int): Frame index to analyze (default: -1, last frame).
        average_over_frames (int): Number of frames to average over for smoother profiles.
        window_size (int): Window size for smoothing packing fraction profiles.
        phrase (str): Common prefix in filenames to identify relevant GSD files.
        N_bins (int): Number of bins for spatial division in profile calculation.
        phase_range (float): Spatial range to define condensed and dilute phase regions.
        r_core (float): Radius of core particles for volume calculations.
    """

    def __init__(self, directory, n_patches, frame=-1, average_over_frames=10, window_size=5, phrase='equil_kT', N_bins=250, phase_range=10, r_core=0.5):
        self.directory = directory
        self.equil_files = [file for file in os.listdir(directory) if file.startswith(phrase)]
        self.kTs = [float(re.search(f'{phrase}(\d+(\.\d+)?)', file).group(1)) for file in self.equil_files]
        self.kTs.sort()
        self.packing_fractions = [Packing_Fraction(os.path.join(directory, f), n_patches, frame, r_core, window_size, N_bins, average_over_frames, phase_range) for f in self.equil_files]
        self._build()

    def _build(self):
        """
        Calculate average packing fractions for condensed and dilute phases across conditions.
        """
        self.dilute_phi = [pf.get_phi_dilute() for pf in self.packing_fractions]
        self.condensed_phi = [pf.get_phi_condensed() for pf in self.packing_fractions]

    def visualise_profiles(self, ax=None):
        """
        Visualize the smoothed packing fraction profiles across all conditions.
        """
        # Create a color gradient for visual differentiation of conditions.
        colors = [matplotlib.cm.coolwarm(i) for i in np.linspace(0, 1, len(self.kTs))]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        # Plot each profile with corresponding color.
        for pf, color in zip(self.packing_fractions, colors):
            pf.draw_phi_profile(ax, color=color, alpha=0.01)  # Adjust alpha as needed

    def get_critical_point(self, m, n):
        """
        Determine the critical point of phase transition using the law of rectilinear diameters.
        (phi_cond - phi_dil)^gamma = d*(1 - T/Tc)
        (phi_cond + phi_dil)/2 = phic + s2*(Tc - T)
        
        Args:
            m (int): Start index in the temperature array for fitting.
            n (int): End index in the temperature array for fitting.
        
        Returns:
            tuple: Contains critical psi (psic), critical temperature (Tc), d and s2 from fitting.
        """
        # Apply linear regression on selected data range to find critical point parameters.
        # Check statistical significance of the regression to ensure reliable results.

        gamma = 3.06 #from 3D Ising model
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.kTs[m:n], (np.array(self.condensed_phi[m:n]) - np.array(self.dilute_phi[m:n]))**gamma)
        if p_value > 0.05:
            raise ValueError("Linear regression for critical point is not statistically significant.")
        d = intercept
        Tc = -(1/slope)*d
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(Tc - self.kTs[m:n], (np.array(self.condensed_phi[m:n]) + np.array(self.dilute_phi[m:n]))/2)
        if p_value > 0.05:
            raise ValueError("Linear regression for phiic is not statistically significant.")
        phic = intercept
        s2 = slope
        
        return phic, Tc, d, s2

    

def get_rel_pos(traj, centr=False, rotation=False, n_core=0, n_patch=0):
    """
    Calculate relative positions of a patch to its core particle in Cartesian and polar coordinates,
    optionally considering rotation.

    Args:
    - traj: Trajectory from a GSD file loaded as a HOOMD-blue trajectory object.
    - centr: If True, centralize the relative positions based on their mean.
    - rotation: If True, adjust for the core's rotation.
    - n_core: Index of the core particle.
    - n_patch: Index of the patch relative to the core particle.

    Returns:
    - Tuple of arrays: (relative positions in Cartesian coordinates, relative positions in polar coordinates).
    """
    N = sum(p.typeid == 0 for p in traj[0].particles)  # Corrected count for core particles
    core_pos = np.array([frame.particles.position[n_core] for frame in traj])
    patch_indices = n_core * 3 + n_patch + np.arange(len(traj)) * N  # Adjusted patch indexing
    patch_pos = np.array([frame.particles.position[idx % len(frame.particles.position)] for idx, frame in zip(patch_indices, traj)])

    # Relative positions considering periodic boundary conditions
    box = traj[0].configuration.box[:3]
    rel_patch_pos = patch_pos - core_pos
    rel_patch_pos -= np.floor(rel_patch_pos / box + 0.5) * box  # Correct periodic boundary adjustment

    if rotation:
        quat_core = np.array([frame.particles.orientation[n_core] for frame in traj])
        rel_patch_quats = quaternion.as_quat_array(np.column_stack((np.zeros(len(quat_core)), rel_patch_pos)))
        core_orientations = quaternion.as_quat_array(quat_core)
        rotated_patch_quats = core_orientations * rel_patch_quats * np.conjugate(core_orientations)
        rel_patch_pos = quaternion.as_float_array(rotated_patch_quats)[:, 1:]

    r = np.linalg.norm(rel_patch_pos, axis=1)
    theta = np.arccos(np.clip(rel_patch_pos[:, 2] / r, -1, 1))  # Clip values to avoid NaNs due to numerical errors
    phi = np.arctan2(rel_patch_pos[:, 1], rel_patch_pos[:, 0])
    rel_patch_pos_polar = np.column_stack((r, theta, phi))

    if centr:
        rel_patch_pos_polar -= np.mean(rel_patch_pos_polar, axis=0)

    return rel_patch_pos, rel_patch_pos_polar


def orthogonal_direction(P1, P2):
    """
    Calculate a normalized direction vector orthogonal to the plane defined by two points on a sphere and the sphere's center.
    
    Args:
    - P1, P2: Coordinates of the two points on the sphere (assumed to be normalized).

    Returns:
    - A normalized vector representing a direction orthogonal to the plane defined by P1, P2, and the sphere's center.
    """
    # Ensure P1 and P2 are normalized
    P1_normalized = P1 / np.linalg.norm(P1)
    P2_normalized = P2 / np.linalg.norm(P2)
    
    # Find a vector orthogonal to both P1 and P2 using the cross product
    orthogonal_vector = np.cross(P1_normalized, P2_normalized)
    
    # Normalize the orthogonal vector to ensure it lies on the unit sphere
    orthogonal_vector_normalized = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    
    # Convert the normalized orthogonal vector back to spherical coordinates
    r = 1  # Unit sphere radius
    theta = np.arccos(orthogonal_vector_normalized[2])  # Inclination
    phi = np.arctan2(orthogonal_vector_normalized[1], orthogonal_vector_normalized[0])  # Azimuth
    
    return orthogonal_vector_normalized, (r, theta, phi)

def haversine(psi1, psi2, th1, th2):
    """
    Calculate the great-circle distance between two points on a sphere given their longitudes and latitudes.

    Args:
    - psi1, th1: Longitude and latitude of the first point in radians.
    - psi2, th2: Longitude and latitude of the second point in radians.

    Returns:
    - The great-circle distance between the two points, in the units of sphere's radius.
    """

    d_th = th2 - th1
    d_psi = psi2 - psi1
    a = np.sin(d_th/2)**2 + np.cos(th1) * np.cos(th2) * (np.sin(d_psi/2)**2)
    distance = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return distance

haversine_vec = np.vectorize(haversine)
