import hoomd
import hoomd.md
import numpy as np
import gsd.hoomd
import itertools
import math
import freud

from typing import Optional

class ForceInitializer:
    """
    Initializes and configures force fields for molecular dynamics simulations using HOOMD-blue.

    Attributes:
        force (hoomd.md.pair.ExpandedMie): Expanded Mie pairwise force for non-bonded interactions.
        bond (hoomd.md.bond.Harmonic): Harmonic potential for bonded interactions.
        angle (hoomd.md.angle.Harmonic): Harmonic potential for angle interactions.
        special_force (hoomd.md.special_pair.LJ): Lennard-Jones potential for special pairwise interactions.
    """

    def __init__(self, buffer: float = 0.4, mode: str = 'shift') -> None:
        """
        Initializes the force fields with specified buffer and computation mode.

        Parameters:
            buffer (float): The buffer distance for the neighbor list. a neighbor list computed on one step can be reused on subsequent steps until a particle moves a distance buffer/2
            mode (str): The mode for computing the pairwise force. See https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html#hoomd.md.pair
        """
        cell = hoomd.md.nlist.Cell(buffer=buffer)
        self.force = hoomd.md.pair.ExpandedMie(nlist=cell, mode=mode)
        self.bond = hoomd.md.bond.Harmonic()
        self.angle = hoomd.md.angle.Harmonic()
        self.special_force = hoomd.md.special_pair.LJ()

    def describe_force(self, p1: str, p2: str, n: float, m: float, eps: float, r: float, r_cut: float, shift_x: bool = False, r_on: Optional[float] = None) -> None:
        """
        Defines the parameters for the Expanded Mie pairwise force between two particle types.

        Parameters:
            p1, p2 (str): Particle types between which the force is applied.
            n, m (float): Exponents in the Mie potential.
            eps (float): Depth of the potential well.
            r (float): Effective radius of the particles.
            r_cut (float): Cutoff radius beyond which the force is not computed.
            shift (bool): Whether to shift the potential to zero at the cutoff.
            r_on (Optional[float]): Distance at which the potential starts to be shifted (if shifting is enabled).
        """
        # The `delta` variable calculates the distance to the potential minimum. When `shift=True`, 
        # this effectively shifts the potential in the x-direction, removing the repulsive part 
        delta = ((n / m) * ((2 * r) ** (n - m))) ** (1 / (n - m)) if shift_x else 0
        self.force.params[(p1, p2)] = {'epsilon': eps, 'sigma': 2 * r, 'n': n, 'm': m, 'delta': -delta}
        self.force.r_cut[(p1, p2)] = 2 * r_cut
        if r_on is not None:
            self.force.r_on[(p1, p2)] = 2 * r_on

    def describe_special_force(self, eps: float, r: float, r_cut: float) -> None:
        """
        Defines the parameters for special Lennard-Jones pairwise interactions.

        Parameters:
            eps (float): Depth of the potential well.
            r (float): Effective radius of the interactions.
            r_cut (float): Cutoff radius beyond which the force is not computed.
        """
        self.special_force.params['intra'] = {'epsilon': eps, 'sigma': 2 * r}
        self.special_force.r_cut['intra'] = 2 * r_cut

    def describe_bond(self, r0: float, k: float = 1) -> None:
        """
        Defines the harmonic bond parameters.

        Parameters:
            r0 (float): Equilibrium distance between bonded particles.
            k (float): Force constant of the bond.
        """
        self.bond.params['core-patch'] = {'k': k, 'r0': r0}

    def describe_angle(self, t0: float, k: float = 1) -> None:
        """
        Defines the harmonic angle parameters.

        Parameters:
            t0 (float): Equilibrium angle in radians.
            k (float): Force constant of the angle.
        """
        self.angle.params['patch-core-patch'] = {'k': k, 't0': t0}


class ParticleInitializer:

    def __init__(self, r_core=0.5, r_patch=0.12, n_patches=0, core_mass=1, is_rigid=False, angles_inter=False, uniform_patches=True, intra=False):
        self.snapshot = gsd.hoomd.Frame()
        self.r_patch = r_patch
        self.r_core = r_core
        self.n_patches = n_patches
        self.core_mass = core_mass
        self.patch_mass = core_mass * ((r_patch**3)/(r_core**3)) 
        self.is_rigid = is_rigid
        self.angles_inter = angles_inter
        self.uniform_patches = uniform_patches
        self.intra = intra

        self._init_patch_positions()
        self._init_inertia_moment()

        if n_patches>0 and is_rigid:
            self._init_rigid()

    def _init_patch_positions(self):
        """Initializes the positions of patches based on the number of patches."""
        if self.n_patches == 3:
            self.patch_positions = self._triangle_patch_positions()
        elif self.n_patches == 4:
            self.patch_positions = self._tetrahedral_patch_positions()
        else:
            raise ValueError(f"Unsupported number of patches: {self.n_patches}")

    def _triangle_patch_positions(self):
        """Calculate positions for three patches in a triangular arrangement."""
        return [[0, self.r_core, 0], 
                [self.r_core * (3**0.5 / 2), -self.r_core / 2, 0],
                [-self.r_core * (3**0.5 / 2), -self.r_core / 2, 0]]

    def _tetrahedral_patch_positions(self):
        """Calculate positions for four patches in a tetrahedral arrangement."""
        return [[self.r_core * (8 / 9)**0.5, 0, -self.r_core / 3],
                [-self.r_core * (2 / 9)**0.5, self.r_core * (2 / 3)**0.5, -self.r_core / 3],
                [-self.r_core * (2 / 9)**0.5, -self.r_core * (2 / 3)**0.5, -self.r_core / 3],
                [0, 0, self.r_core]]

    def _init_inertia_moment(self):
        """
        Initializes the moment of inertia for both patch and core particles.
        """
        # Moment of inertia for a single patch based on the solid sphere formula (2/5)*m*r^2
        self.I_patch = (2/5) * self.patch_mass * self.r_patch**2 * np.identity(3)
    
        # Moment of inertia for a single core based on the solid sphere formula (2/5)*m*r^2
        self.I_core = (2/5) * self.core_mass * self.r_core**2 * np.identity(3)
    
        # If the system is rigid and has patches, adjust the core's moment of inertia to account for the patches
        if self.n_patches > 0 and self.is_rigid:
            for position in self.patch_positions:
                # Calculate the contribution of this patch to the core's moment of inertia
                # using the parallel axis theorem: I = I_cm + m*d^2, where I_cm is the moment
                # of inertia about the center of mass, m is the mass, and d is the distance
                # from the axis to the center of mass.
                self.I_core += self.I_patch + self.patch_mass * np.dot(np.array(position), np.array(position))


    def _init_rigid(self):
        """
        Initialize the rigid body by diagonalizing the inertia tensor and setting up
        the constituent particles.
        """
        self._diagonalisation()
        self.rigid = hoomd.md.constrain.Rigid()
        self.rigid.body['core'] = {
            "constituent_types": ['patch']*self.n_patches,
            "positions": self.patch_positions,
            "orientations": np.tile([1.0, 0.0, 0.0, 0.0], (self.n_patches, 1))
        }
    
    def _diagonalisation(self):
        """
        Diagonalize the inertia tensor of the core, rotate the patch positions
        accordingly, and update the inertia tensor to its diagonal form.
        """
        if not self.I_core.shape == (3, 3):
            raise ValueError("I_core must be a 3x3 matrix.")
        
        I_diagonal, E_vec = np.linalg.eig(self.I_core)
        self.patch_positions = np.dot(E_vec, np.array(self.patch_positions).T).T
        self.I_core = np.diag(I_diagonal)


    def _set_common_attributes(self, positions, box_dims):
        """
        Set common attributes for the simulation based on input positions and box dimensions.
        First initialise cores, then add patches, if there are some

        Parameters:
        - positions: Array of particle positions (only cores).
        - box_dims: Dimensions of the simulation box.

        This method initializes the particle system, sets particle types, assigns masses,
        and if necessary, adds and connects patches to each core particle.
        """
        N_particles = len(positions)
        self.N_cores = N_particles
        self.snapshot.particles.N = N_particles
        self.snapshot.particles.position = positions
        self.snapshot.particles.typeid = np.zeros(N_particles, dtype=int).tolist()
        self.snapshot.configuration.box = box_dims
        self.snapshot.particles.moment_inertia = np.tile([self.I_core[0, 0], self.I_core[1, 1], self.I_core[2, 2]], (N_particles, 1)).tolist()

        # Set particle types based on uniformity of patches
        if self.uniform_patches:
            self.snapshot.particles.types = ['core', 'patch']
        else:
            self.snapshot.particles.types = ['core'] + ['patch' + str(i) for i in range(self.N_cores)]

        self.snapshot.particles.mass = np.full(N_particles, self.core_mass).tolist()

        #add patches to particles
        if self.n_patches > 0:
            self._add_patches()
            self._connect_patches()

    def _add_patches(self):
        # Calculate the total number of patches to add
        total_patches = self.N_cores * self.n_patches
    
        # Adjust type IDs for patches
        if self.uniform_patches:
            new_type_ids = [1] * total_patches  # All patches have the same type
        else:
            new_type_ids = np.repeat(np.arange(1, self.N_cores + 1), self.n_patches).tolist()  # Unique type for each core's patches
    
        # Update particle properties
        self.snapshot.particles.typeid += new_type_ids
        self.snapshot.particles.mass += [self.patch_mass] * total_patches
        self.snapshot.particles.N += total_patches
        self.snapshot.particles.moment_inertia += [[self.I_patch[0, 0], self.I_patch[1, 1], self.I_patch[2, 2]]] * total_patches

        # Calculate new positions for all patches
        core_positions = np.array(self.snapshot.particles.position)
        patch_positions = np.array(self.patch_positions)
        # For each core, calculate positions of its patches and add them to the list
        new_positions = np.vstack([core_pos + patch_pos for core_pos in core_positions for patch_pos in patch_positions])
        # Append new positions to the snapshot
        self.snapshot.particles.position.extend(new_positions.tolist())

    def _connect_patches(self):

        n_interactions = int(self.N_cores * self.n_patches * (self.n_patches - 1) / 2)
        interaction_groups = []
        bond_groups = []
        for core_id in range(self.N_cores):
            patch_ids = range(core_id * self.n_patches + self.N_cores, (core_id + 1) * self.n_patches + self.N_cores)
            for i, patch_id in enumerate(patch_ids):
                bond_groups.append([core_id, patch_id])
                for next_patch_id in patch_ids[i + 1:]:
                    interaction_groups.append([patch_id, core_id, next_patch_id] if self.angles_inter else [patch_id, next_patch_id])

        total_bonds = self.N_cores * self.n_patches
        self.snapshot.bonds.N = total_bond
        self.snapshot.bonds.types = ['core-patch']
        self.snapshot.bonds.typeid = [0] * total_bonds
        self.snapshot.bonds.group = bond_groups

        if self.angles_inter:
            self.snapshot.angles.N = n_interactions
            self.snapshot.angles.types = ['patch-core-patch']
            self.snapshot.angles.typeid = [0] * n_interactions
            self.snapshot.angles.group = [group for group in interaction_groups if len(group) == 3]

        if self.intra:
            self.snapshot.pairs.N = n_interactions
            self.snapshot.pairs.types = ['intra']
            self.snapshot.pairs.typeid = [0] * n_interactions
            # For intra, adjust groups to only include patch-patch pairs
            self.snapshot.pairs.group = [group for group in interaction_groups if len(group) == 2]


    def save_to_gsd(self, name, seed=0):
        """
        Saves the current snapshot to a GSD file. Handles both simple and rigid-body systems.

        Parameters:
        - name (str): Filename for the GSD file.
        - seed (int, optional): Seed for the random number generator. Defaults to 0.
        """
        # Error checking for filename
        if not name.endswith('.gsd'):
            raise ValueError("Filename must end with '.gsd'.")

        device =  hoomd.device.CPU()

        if self.n_patches == 0 or not self.is_rigid:
            with gsd.hoomd.open(name=name, mode='wb') as f:
                f.append(self.snapshot)
        else:
            try:
                # Create a simulation with the specified device and seed
                sim = hoomd.Simulation(device=device, seed=seed)
                sim.create_state_from_snapshot(self.snapshot)
                # Create rigid bodies and save the state
                self.rigid.create_bodies(sim.state)
                hoomd.write.GSD.write(state=sim.state, mode='wb', filename=name)
            except Exception as e:
                print(f"Error saving to GSD: {e}")
                # Handle or log error as appropriate
            finally:
                # Cleanup resources, if any
                del sim

    def init_lattice(self, lattice_type='sc', num_units=10, spacing=1.1, uniform=True):
        """
        Initializes a particle system into a specified lattice configuration.

        This method supports simple cubic (SC), face-centered cubic (FCC), and hexagonal close-packed (HCP) lattices.
        
        Parameters:
        - lattice_type (str): The type of lattice to initialize ('sc', 'fcc', or 'hcp').
        - num_units (int): The number of unit cells along each dimension for 'sc' and 'hcp', or the number of replicas for 'fcc'.
        - spacing (float): Multiplier to adjust the distance between particles, with 1 being the default close-packed spacing.
        - uniform (bool): If True (default for 'sc'), the lattice is filled uniformly; otherwise, particles are added up to `num_units`.
        """

        if lattice_type == 'sc':
            N_particles_in_row = math.ceil(num_units ** (1 / 3))
            L = N_particles_in_row * 2 * self.r_core * spacing
            x = np.linspace(-L / 2, L / 2, N_particles_in_row, endpoint=False)
            positions = list(itertools.product(x, x, x))
            if not uniform:
                positions = positions[:num_units]
            box_dims = [L + 1, L + 1, L + 1, 0, 0, 0]

        elif lattice_type == 'fcc':
            fcc = freud.data.UnitCell.fcc()
            fcc_system = fcc.generate_system(num_replicas=num_units, scale=self.r_core * 2 ** (1.5))
            positions = fcc_system[1]
            box_dims = [fcc_system[0].Lx, fcc_system[0].Ly, fcc_system[0].Lz, fcc_system[0].xy, fcc_system[0].xz, fcc_system[0].yz]

        elif lattice_type == 'hcp':
            positions = []
            for i in range(num_units):
                for j in range(num_units):
                    for k in range(num_units):
                        pos = [(2 * i + ((j + k) % 2)) * self.r_core,
                               (j + (k % 2) / 3) * np.sqrt(3) * self.r_core,
                               (2 * np.sqrt(6) * k / 3) * self.r_core]
                        positions.append(pos)
            positions = np.array(positions)
            center = np.mean(positions, axis=0)
            positions -= center  # Center the positions
            Lx, Ly, Lz = np.ptp(positions, axis=0) + 2 * self.r_core * spacing
            box_dims = [Lx, Ly, Lz, 0, 0, 0]

        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

        self._set_common_attributes(positions, box_dims)

    

    
    
