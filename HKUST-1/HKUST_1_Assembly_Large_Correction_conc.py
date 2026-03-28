MAX_DEPTH = 3
# Adjusting the file reading function with the new understanding of the connectivity data format

import itertools
import random
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import copy
import weakref
from collections import deque
# from memory_profiler import profile


class RandomizedSet:
    def __init__(self):
        self.items = []          # 动态列表，支持 O(1) 随机访问
        self.item_to_index = {}  # 哈希字典，记录每个元素在列表中的索引

    def add(self, val):
        if val in self.item_to_index:
            return False
        self.item_to_index[val] = len(self.items)
        self.items.append(val)
        return True

    def remove(self, val):
        if val not in self.item_to_index:
            return False
        index = self.item_to_index[val]
        last_item = self.items[-1]
        self.items[index] = last_item
        self.item_to_index[last_item] = index
        self.items.pop()
        del self.item_to_index[val]
        return True

    def get_random(self):
        return random.choice(self.items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
    
    def __contains__(self, val):
        return val in self.item_to_index
    
    def to_list(self):
        return self.items.copy()
    
    def update(self, iterable):
        for item in iterable:
            self.add(item)


SUPERIMPOSE_THRESHOLD = 0.1
         
def read_mol_file_v5(file_path):
    elements = []
    coordinates = []
    connectivity_map = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # is_coordinate_block = True  # Indicates if we are reading the coordinate block
        for line in lines[4:]:  # Skip the first four lines (header)
            if 'M  END' in line:
                break  # End of file data
            parts = line.split()
            if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():  # Connectivity map line
                atom1, atom2, bond_order = int(parts[0]), int(parts[1]), int(parts[2])
                connectivity_map.append((atom1-1, atom2-1, bond_order))
            elif len(parts) > 6 and parts[3].isalpha():  # 3D coordinate line
                x, y, z, atom_type = np.float32(parts[0]), np.float32(parts[1]), np.float32(parts[2]), parts[3]
                elements.append(atom_type)
                coordinates.append(np.array([x, y, z]))
    
    coordinates = np.array(coordinates)

    # search for carboxylates for connection 先确定与O连接的C，与两个O连接的C，即为羧酸基团，找到羧酸中心，计算最远原子到中心的距离
    oxygen_atoms = np.where(np.array(elements) == 'O')[0]
    carbon_atoms = np.where(np.array(elements) == 'C')[0]
    # print(oxygen_atoms,carbon_atoms)
    # Create a lookup table for connections involving carbon and oxygen atoms
    connections = {}
    for conn in connectivity_map:
        a, b, _ = conn
        if a in carbon_atoms and b in oxygen_atoms:
            connections.setdefault(a, []).append(b)
        elif b in carbon_atoms and a in oxygen_atoms:
            connections.setdefault(b, []).append(a)
    carboxylate_indices = []
    # Identify carboxylates based on connectivity
    for carbon_index, oxygen_indices in connections.items():
        if len(oxygen_indices) == 2:  # A carboxylate requires two bonded oxygen atoms
            carboxylate_indices.append(np.array([carbon_index] + oxygen_indices))
    center = np.mean(coordinates,axis=0)
    radius = np.max(np.linalg.norm(coordinates - center, axis=1)) +3.0
        
    return elements, coordinates, carboxylate_indices, center, radius



# Implementing the logic to identify carboxylates in the Zr6 and BTB classes using the connectivity data 
    # Function to rotate the carboxylate to align it to the reference_carboxylate旋转时羧酸盐对齐

class Carboxylate:
    def __init__(self, atom_indexes, entity):
        self.atom_indexes = atom_indexes
        # self.belonging_entity = weakref.proxy(entity)
        self.belonging_entity = entity
        if entity.entity_type == 'Zr':
            self.carboxylate_type = 'formate'#AA
        else:
            self.carboxylate_type = 'benzoate'
    
    # Function to test if the carboxylate is superimposed with another carboxylate of different types 判断羧酸距离是否合适能进行连接
    def carboxylates_superimposed(self, carboxylate2, threshold=SUPERIMPOSE_THRESHOLD):
        """
        Check if two carboxylates are superimposed.
        :param carboxylate1: First carboxylate to compare.
        :param carboxylate2: Second carboxylate to compare.
        :param threshold: Distance threshold for considering carboxylates superimposed.
        :return: True if carboxylates are superimposed, False otherwise.
        """
        # Avoid comparing carboxylate of the same type
        if self.carboxylate_type == carboxylate2.carboxylate_type:
            return False
        
        threshold_sq = 3*threshold**2 #不懂
        #compare the carbon atoms

        diff = (self.belonging_entity.coordinates[self.atom_indexes[0]] - carboxylate2.belonging_entity.coordinates[carboxylate2.atom_indexes[0]]).reshape(1,-1)
        if np.dot(diff,diff.T) > threshold_sq:
            # print('carbon atoms not matched')
            return False
        
        # Compare oxygen atoms
        for o1 in self.atom_indexes[1:]:
            for o2 in carboxylate2.atom_indexes[1:]:
                diff = (self.belonging_entity.coordinates[o1] - carboxylate2.belonging_entity.coordinates[o2]).reshape(1,-1)
                if np.dot(diff,diff.T) <= threshold_sq:   
                    return True
        # print('No pair within threshold')
        
        return False


    def calculate_rotation_translation_matrix(self, reference_carboxylate):
        """
        Calculate a rotation matrix that aligns points_from to points_to.
        :param reference_carboxylate: The carboxylate to align to.
        :return: 4x4 rotation and translation matrix.
        """
        # Efficient construction of points_to_align
        points_from = self.belonging_entity.coordinates[self.atom_indexes]

        # Efficient construction of points_reference
        points_to = reference_carboxylate.belonging_entity.coordinates[reference_carboxylate.atom_indexes]
        
        # Calculate centroids
        centroid_from = np.mean(points_from, axis=0)
        centroid_to = np.mean(points_to, axis=0)  

        # Center the points
        internal_from = points_from - centroid_from
        internal_to = points_to - centroid_to
        
        # Compute the covariance matrix
        H = internal_from.T @ internal_to
        
        # Singular Value Decomposition
        U, _, Vt = svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # # Ensure a right-handed coordinate system
        # if np.linalg.det(R) < 0:
        #     Vt[-1, :] *= -1
        #     R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_to - R @ centroid_from
        
        # Construct the 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t 

        return transformation_matrix
    
    # Helper function for visualizing carboxylates
    def visualize_carboxylates(self, ref_carboxylate, title='Carboxylate Alignment'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Extract and plot coordinates for the first carboxylate
        coords1 = self.belonging_entity.coordinates[self.atom_indexes]
        xs1, ys1, zs1 = zip(*coords1)
        ax.scatter(xs1, ys1, zs1, c='blue', marker='o', label='Carboxylate 1')
    
        # Extract and plot coordinates for the second carboxylate
        coords2 = ref_carboxylate.belonging_entity.coordinates[ref_carboxylate.atom_indexes]
        xs2, ys2, zs2 = zip(*coords2)
        ax.scatter(xs2, ys2, zs2, c='red', marker='o', label='Carboxylate 2')
    
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(title)
        ax.legend()
        plt.show()


class SBU_Entity:
    """
    A mixin class providing 4D transformation capabilities for rotation and translation.
    """
    def __init__(self, elements, coordinates, carboxylate_indices, center, radius):
        self.elements = elements
        self.carboxylates = [Carboxylate(carboxylate_index,self) for carboxylate_index in carboxylate_indices]
        self.coordinates = coordinates.copy()
        self.center = center.copy()
        self.radius = radius
        self.connected_entities = []
        self.kdtree = None

    def create_kdtree(self):
        """
        Create a KDTree for the entity's atoms.
        """
        if self.kdtree is None:
            # Check if coordinates exist and are valid
            if not hasattr(self, 'coordinates') or self.coordinates is None:
                return
            if len(self.coordinates) == 0:
                return
            try:
                self.kdtree = cKDTree(self.coordinates)
            except Exception as e:
                print(f"Warning: Failed to create KDTree: {e}")
                self.kdtree = None
    # -----------------------------modified--------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # kdtree is a derived cache, exclude from persistence to prevent recursion/serialization issues
        state['kdtree'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Maintain lazy rebuild after loading
        self.kdtree = None
    # ----------------------------------------------------------------
    
    def apply_transformation(self, matrix):
        """
        Apply a 4D transformation matrix to the coordinates of the object.
        :param matrix: 4x4 numpy array representing the transformation matrix.
        """

        coords = np.hstack((self.coordinates, np.ones((self.coordinates.shape[0], 1))))
        transformed_coords = (matrix @ coords.T).T
        self.coordinates = transformed_coords[:, :3]

        # Update the center
        self.center = np.mean(self.coordinates,axis=0)

        # Invalidate the old KD-tree because coordinates changed
        self.kdtree = None

        #######################################################################################many imortant things to update
    
    # Helper function for visualization
    def visualize_entities(self, Ref_entity, title='Molecular Assembly'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = {'To Align': 'blue', 'Reference': 'red'}
    
        xs, ys, zs = zip(*[coord for coord in self.coordinates])
        ax.scatter(xs, ys, zs, c=colors['To Align'], marker='o', label='To Align')
        xs, ys, zs = zip(*[coord for coord in Ref_entity.coordinates])
        ax.scatter(xs, ys, zs, c=colors['Reference'], marker='o', label='Reference')        
    
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(title)
        ax.legend()
        plt.show()


    
    # Alignment function according to carboxylate coordinate
    def align_carboxylates(self, reference_carboxylate, visualize_steps=False):
        carboxylate_to_align = random.choice(self.carboxylates)
    
        if visualize_steps:
            carboxylate_to_align.visualize_carboxylates(reference_carboxylate, 'Before Alignment')
            self.visualize_entities(reference_carboxylate.belonging_entity, title='Before Alignment')
    
        # Calculate rotation matrix for plane alignment
        rotation_translation_matrix = carboxylate_to_align.calculate_rotation_translation_matrix(reference_carboxylate)

        # print(rotation_translation_matrix)
        self.apply_transformation(rotation_translation_matrix)
        del rotation_translation_matrix
        
        # Visualize after rotation
        if visualize_steps:
            carboxylate_to_align.visualize_carboxylates(reference_carboxylate, 'After Alignment')
            self.visualize_entities(reference_carboxylate.belonging_entity, title='After Alignment')

        return carboxylate_to_align

PRELOADED_DATA = {
    'Zr6_AA': read_mol_file_v5('/home/syh/MOL/mol_growth/mol/Cu2.mol'),
    'Zr12_AA': read_mol_file_v5('/home/syh/MOL/mol_growth/mol/Zr12_AA.mol'),
    'BDC': read_mol_file_v5('/home/syh/MOL/mol_growth/mol/BTC.mol')
}

class Zr6_AA(SBU_Entity):
    def __init__(self):
        self.entity_type = 'Zr'
        self.entity_subtype = 0
        # elements, coordinates, carboxylate_indices, center, radius = PRELOADED_DATA['Zr6']
        super().__init__(PRELOADED_DATA['Zr6_AA'][0], PRELOADED_DATA['Zr6_AA'][1], PRELOADED_DATA['Zr6_AA'][2], PRELOADED_DATA['Zr6_AA'][3], PRELOADED_DATA['Zr6_AA'][4])
        
        
class Zr12_AA(SBU_Entity):
    def __init__(self):
        self.entity_type = 'Zr'
        self.entity_subtype = 1
        # elements, coordinates, carboxylate_indices, center, radius = PRELOADED_DATA['Zr12']
        # super().__init__(elements, coordinates, carboxylate_indices, center, radius)
        super().__init__(PRELOADED_DATA['Zr12_AA'][0], PRELOADED_DATA['Zr12_AA'][1], PRELOADED_DATA['Zr12_AA'][2], PRELOADED_DATA['Zr12_AA'][3], PRELOADED_DATA['Zr12_AA'][4])
        
        
class BDC(SBU_Entity):
    def __init__(self):
        self.entity_type = 'Ligand'
        self.entity_subtype = 0#不懂
        # elements, coordinates, carboxylate_indices, center, radius = PRELOADED_DATA['BTBa']
        # super().__init__(elements, coordinates, carboxylate_indices, center, radius)
        super().__init__(PRELOADED_DATA['BDC'][0], PRELOADED_DATA['BDC'][1], PRELOADED_DATA['BDC'][2], PRELOADED_DATA['BDC'][3], PRELOADED_DATA['BDC'][4])

# class BTBb(SBU_Entity):
#     def __init__(self):
#         self.entity_type = 'Ligand'
#         self.entity_subtype = 1
#         # elements, coordinates,  carboxylate_indices, center, radius = PRELOADED_DATA['BTBb']
#         # super().__init__(elements, coordinates, carboxylate_indices, center, radius)
#         super().__init__(PRELOADED_DATA['BTBb'][0], PRELOADED_DATA['BTBb'][1], PRELOADED_DATA['BTBb'][2], PRELOADED_DATA['BTBb'][3], PRELOADED_DATA['BTBb'][4])


class Assembly:
    """
    Class to represent an assembly of Zr6 and BTB entities.
    """
    def __init__(self, initial_entity, ZR6_PERCENTAGE, ENTROPY_GAIN, BUMPING_THRESHOLD):
        # Add a new judgement to distinguish initial entity is assembly or not
        if isinstance(initial_entity, Assembly):
            # 直接使用对象，避免 deepcopy 破坏引用关系（特别是 belonging_entity）
            self.__dict__ = initial_entity.__dict__
            self.ZR6_PERCENTAGE = ZR6_PERCENTAGE
            self.ENTROPY_GAIN = ENTROPY_GAIN
            self.BUMPING_THRESHOLD = BUMPING_THRESHOLD
            return
        else:
            self.entities = RandomizedSet()            
            self.entities.add(initial_entity)           
            self.free_carboxylates = RandomizedSet() 
            self.free_carboxylates.update(initial_entity.carboxylates)
            self.MC_free_carboxylates = RandomizedSet()  # A list of free carboxylates on metal cluster entity
            self.Linker_free_carboxylates = RandomizedSet()  # A list of free carboxylates on linker entity
            if initial_entity.entity_type == 'Zr':
                self.MC_free_carboxylates.update(initial_entity.carboxylates)
            else:
                self.Linker_free_carboxylates.update(initial_entity.carboxylates)
            self.linked_carboxylate_pairs = RandomizedSet()
            self.ready_to_connect_carboxylate_pairs = RandomizedSet()
            self.pair_index = {} # Dictionary to store the index of the pairs in the carboxylate_pairs list

            self.ROI_prepare()
            self.node_mapping = {}  # Dictionary to store nodes to different ranges in space. Devide the space to cubes of ROI_range**3 in size. The keys are the coordinates of these cubes.
            cube_index = tuple(np.floor(initial_entity.center/self.ROI_range).astype(int))  # The cube index in which the center of the entity is located
            self.node_mapping[cube_index] = [initial_entity]

            self.time_for_align_new_entity = 0
            self.time_for_creating_new_entity =0
            self.time_for_find_connection = 0
            self.time_for_finding_ROI = 0
            self.time_for_judge_bumping = 0
            self.time_for_is_too_close = 0
            self.time_for_add_new_entity = 0

            self.time_for_cut_a_bond = 0
            self.time_for_remove_entities = 0

            self.ZR6_PERCENTAGE = ZR6_PERCENTAGE
            self.ENTROPY_GAIN = ENTROPY_GAIN
            self.BUMPING_THRESHOLD = BUMPING_THRESHOLD

    def ROI_prepare(self):
        # Set four entities Zr6, Zr12, BTBa, BTBb
        entities = [Zr6_AA(), Zr12_AA(), BDC()]
        # Calculate their radius and pick the maximum
        radii = [entity.radius for entity in entities]
        max_radius = max(radii)
        # Use twice the maximum radius as the ROI range
        self.ROI_range = 2 * max_radius

    def next_thing_to_do(self, FORMATE_BENZOATE_RATIO, entropy_correction_table,Zr6_conc_adding_probability):
        """
        Decide the next step in the assembly process based on the current state.
        :param FORMATE_BENZOATE_RATIO: Ratio affecting linked carboxylates.
        :param SBU_PRESENT_ORIGINAL_RATIO: Ratio affecting external SBU addition.
        :return: Tuple (action, selected_carboxylate, growth_total).
        """
        num_entity = len(self.entities)
        D_ENTROPY_ZR6_BTB = 3.6 # the entropy difference between Zr6 and BTB in the unit of R
        D_ENTROPY_ZR12_BTB = 3.6 # the entropy difference between Zr12 and BTB in the unit of R

        if num_entity >= 2:
            # Calculate relevant quantities
            num_free_carboxylate_sites = len(self.free_carboxylates)     
            num_internal_to_be_link = len(self.ready_to_connect_carboxylate_pairs)       
            num_linked = len(self.linked_carboxylate_pairs)                              
            
            # Growth total considering external addition, internal linking, and removal of a bond
            growth_total = (num_free_carboxylate_sites + num_internal_to_be_link*entropy_correction_table[num_entity] + num_linked*FORMATE_BENZOATE_RATIO)
            
            # Calculate probabilities for each event
            probability_adding_external = num_free_carboxylate_sites/growth_total
            # print('p_adding',probability_adding_external)
            probability_linking_internal = num_internal_to_be_link*entropy_correction_table[num_entity]/growth_total
            # print('p_internal_linking',probability_linking_internal)
            # probability_removal = num_linked*FORMATE_BENZOATE_RATIO/growth_total
            # print('p_removal',probability_removal)

            # Decide the next step based on the probabilities
            random_select = random.random()
            if random_select < probability_adding_external:
                # external SBU addition
                # selected_carboxylate = self.growth_site_selection()
                # selected_carboxylate = random.choice(self.free_carboxylates)
                # selected_carboxylate = self.free_carboxylates.get_random()  ### Modified   
                # if selected_carboxylate.belonging_entity.entity_type == 'Zr':
                #     if selected_carboxylate.belonging_entity.entity_subtype == 0:
                #         if random.random() < 2*Zr6_conc_adding_probability-0.5:
                #             selected_carboxylate = self.free_carboxylates.get_random() #use the opportunity to reselect the carboxylate to consider the difference in entropy
                #     else:
                #         if random.random() < 1.5-2*Zr6_conc_adding_probability:
                #             selected_carboxylate = self.free_carboxylates.get_random()
                #------------ modified(please check again!)--------------------
                if random.random() < Zr6_conc_adding_probability:
                    # Add Zr6
                    if len(self.Linker_free_carboxylates) > 0:
                        selected_carboxylate = self.Linker_free_carboxylates.get_random()
                    else:
                        selected_carboxylate = self.free_carboxylates.get_random()
                else:
                    # Add BTB
                    if len(self.MC_free_carboxylates) > 0:
                        selected_carboxylate = self.MC_free_carboxylates.get_random()
                    else:
                        selected_carboxylate = self.free_carboxylates.get_random()
                #------------ modified(please check again!)--------------------

                # if random.random() < SBU_PRESENT_ORIGINAL_RATIO:
                #     action = 1
                # else:
                #     action = 2
                return 1, selected_carboxylate, growth_total, None
            
            elif random_select < probability_adding_external + probability_linking_internal:
                # internal linking
                # Clean up invalid ready pairs before selection
                invalid_pairs = []
                for pair in self.ready_to_connect_carboxylate_pairs:
                    if pair and len(pair) == 2:
                        c1, c2 = pair
                        if (not hasattr(c1, 'belonging_entity') or c1.belonging_entity is None or
                            not hasattr(c2, 'belonging_entity') or c2.belonging_entity is None):
                            invalid_pairs.append(pair)
                # Remove invalid pairs
                for invalid_pair in invalid_pairs:
                    self.ready_to_connect_carboxylate_pairs.remove(invalid_pair)
                    # Also try reverse order
                    reverse_pair = (invalid_pair[1], invalid_pair[0])
                    if reverse_pair in self.ready_to_connect_carboxylate_pairs:
                        self.ready_to_connect_carboxylate_pairs.remove(reverse_pair)
                    # Clean up pair_index
                    for carb in invalid_pair:
                        if carb in self.pair_index:
                            self.pair_index.pop(carb, None)
                
                if len(self.ready_to_connect_carboxylate_pairs) == 0:
                    # No valid ready pairs, fall back to external addition
                    if random.random() < Zr6_conc_adding_probability:
                        if len(self.Linker_free_carboxylates) > 0:
                            selected_carboxylate = self.Linker_free_carboxylates.get_random()
                        else:
                            selected_carboxylate = self.free_carboxylates.get_random()
                    else:
                        if len(self.MC_free_carboxylates) > 0:
                            selected_carboxylate = self.MC_free_carboxylates.get_random()
                        else:
                            selected_carboxylate = self.free_carboxylates.get_random()
                    return 1, selected_carboxylate, growth_total, None
                
                # selected_pair = random.choice(self.ready_to_connect_carboxylate_pairs)   
                selected_pair = self.ready_to_connect_carboxylate_pairs.get_random()  ### Modified 
                return 0, None, growth_total, selected_pair
            
            else:
                # removal of a linked bond
                # selected_pair = random.choice(self.linked_carboxylate_pairs)    
                selected_pair = self.linked_carboxylate_pairs.get_random()  ### Modified
                return -1, None, growth_total, selected_pair
            
        else:
            # selected_carboxylate = random.choice(self.free_carboxylates)  
            selected_carboxylate = self.free_carboxylates.get_random()  ### Modified
            return 1, selected_carboxylate, len(self.free_carboxylates), None

    def link_internal_carboxylate(self, selected_pair):
        """
        Link an internal carboxylate to its pair and update all relevant structures.
        :param selected_carboxylate: The carboxylate to link.
        """
        # Validate that both carboxylates have valid belonging_entity
        c1, c2 = selected_pair
        if (not hasattr(c1, 'belonging_entity') or c1.belonging_entity is None or
            not hasattr(c2, 'belonging_entity') or c2.belonging_entity is None):
            # Invalid pair: clean it up and skip
            if selected_pair in self.ready_to_connect_carboxylate_pairs:
                self.ready_to_connect_carboxylate_pairs.remove(selected_pair)
            # Also try reverse order
            reverse_pair = (c2, c1)
            if reverse_pair in self.ready_to_connect_carboxylate_pairs:
                self.ready_to_connect_carboxylate_pairs.remove(reverse_pair)
            # Clean up pair_index
            for carb in selected_pair:
                if carb in self.pair_index:
                    self.pair_index.pop(carb, None)
            # Skip this invalid pair
            return
        
        # Update linked carboxylates and pairs
        # self.linked_carboxylate_pairs.append(selected_pair)
        self.linked_carboxylate_pairs.add(selected_pair)  ### Modified

        # Remove the pair and individual carboxylates from ready-to-connect lists
        self.ready_to_connect_carboxylate_pairs.remove(selected_pair)  ### Modified                                            

        # Update connection properties for the selected pair
        for carboxylate in selected_pair:
            the_other_carboxylate = selected_pair[0] if carboxylate == selected_pair[1] else selected_pair[1]
            carboxylate.belonging_entity.connected_entities.append(the_other_carboxylate.belonging_entity)

    def grow_one_step(self, selected_carboxylate):
        """
        Perform one step of the growth process by deciding on the new entity to add,
        aligning it, and connecting it to the assembly.
        :param selected_carboxylate: The carboxylate where the growth occurs.
        """
        start_time = time.time()
        # Decide on the new entity type based on the carboxylate type
        if selected_carboxylate.carboxylate_type == 'benzoate':
            new_entity = Zr6_AA() if random.random() < self.ZR6_PERCENTAGE else Zr12_AA()
        else:
            new_entity = BDC() # Create a new Zr6 or BTB entity
            # print('preparing new entity ',new_entity)
        self.time_for_creating_new_entity += (time.time()-start_time)   

        # Align the new entity to the selected carboxylate
        start_time = time.time()
        to_be_link_carboxylate_on_new_entity = new_entity.align_carboxylates(selected_carboxylate, visualize_steps=False)
        self.time_for_align_new_entity += (time.time()-start_time)

        # Find potential connections for the new entity
        start_time = time.time()
        Is_bumping, additional_to_be_connect_carboxylate_pairs, additional_to_be_connect_carboxylate_on_new_entity = self.find_potential_connection(new_entity, selected_carboxylate)
        self.time_for_find_connection += (time.time()-start_time)
        # print('Is_bumping:',Is_bumping)   

        # Add the new entity if no bumping issues
        start_time = time.time()
        if not Is_bumping:
            self.add_entity(selected_carboxylate, new_entity, to_be_link_carboxylate_on_new_entity, additional_to_be_connect_carboxylate_pairs, additional_to_be_connect_carboxylate_on_new_entity)
        else:
            for carboxylate in new_entity.carboxylates:
                carboxylate.belonging_entity = None
            del new_entity
        self.time_for_add_new_entity += (time.time()-start_time)


    def kdtree_method(self, new_entity, other_entity):
        # Build (or reuse) the KDTree for the entity if not already built

        # Check for None entities first
        if new_entity is None or other_entity is None:
            return False  # No collision if either entity is None
        
        # Check if entities have required attributes
        if not hasattr(new_entity, 'create_kdtree') or not hasattr(other_entity, 'create_kdtree'):
            return False
        
        start_time = time.time()
        new_entity.create_kdtree()
        other_entity.create_kdtree()
        kdtree_new = new_entity.kdtree
        kdtree_other = other_entity.kdtree
        # Additional safety check
        if kdtree_new is None or kdtree_other is None:
            self.time_for_is_too_close += (time.time()-start_time)
            return False
        # Query all new_entity points at once
        results = kdtree_new.query_ball_tree(kdtree_other, r=self.BUMPING_THRESHOLD)
        # If any result list is nonempty, there is a bump.
        self.time_for_is_too_close += (time.time()-start_time)
        output = any(len(res) > 0 for res in results)
        # del results
        return output

    def find_potential_connection(self, new_entity, selected_carboxylate): 
        """
        Find potential connections for a new entity and determine if there are any bumps or additional connections.
        :param new_entity: The new entity to check.
        :param selected_carboxylate: The selected carboxylate for connection.
        :return: Tuple (Is_bumping, to_be_link_carboxylate_on_new_entity, additional_to_be_connect_carboxylate_pairs,
                to_be_link_entity, additional_to_be_connect_entity).
        """
        start_time = time.time()
        additional_to_be_connect_carboxylate_pairs = []
        additional_to_be_connect_carboxylate_on_new_entity = []
        to_be_link_entity = selected_carboxylate.belonging_entity 
        # print('to_be_link_entity:',to_be_link_entity)

        Is_bumping = False

        # Find potential connections for the new entity

        # search for nearby entities ROI
        start_time = time.time()
        if len(self.entities) > 30:
            cube_index = tuple(np.floor(new_entity.center/self.ROI_range).astype(int))
            possible_ranges = list(itertools.product(range(cube_index[0]-1, cube_index[0]+2), 
                                    range(cube_index[1]-1, cube_index[1]+2), 
                                    range(cube_index[2]-1, cube_index[2]+2)))
            entity_of_interest = []
            for idx in possible_ranges:
                    if idx in self.node_mapping:
                        entity_of_interest.extend(self.node_mapping[idx])
            # print('entity_of_interest:',entity_of_interest)
            # print('to_be_link_entity:',to_be_link_entity)
            entity_of_interest.remove(to_be_link_entity)

        else:
            # entity_of_interest = self.entities.copy()
            entity_of_interest = self.entities.to_list()  ### Modified
            # print('entity_of_interest:',entity_of_interest)
            # print('to_be_link_entity:',to_be_link_entity)
            entity_of_interest.remove(to_be_link_entity)                             
        self.time_for_finding_ROI += (time.time()-start_time)

        # search for nearby entities ROI
        if len(entity_of_interest) > 0:
            start_time = time.time()
            entity_centers = np.array([entity.center for entity in entity_of_interest])
            entity_radius = np.array([entity.radius for entity in entity_of_interest])
            distances = np.linalg.norm(entity_centers - new_entity.center, axis=1)
            mask = distances <= (entity_radius + new_entity.radius)
            # mask = distances <= self.ROI_range
            indices = np.where(mask)[0]
            entity_of_interest = [entity_of_interest[i] for i in indices]
            # distances = distances[mask]
            self.time_for_finding_ROI += (time.time()-start_time)

            # Check for additional connections and bumping
            start_time = time.time()
            Is_bumping = False
            for entity in entity_of_interest:
                Is_bumping = True
                if  entity.entity_type != new_entity.entity_type:
                    for carboxylate1 in entity.carboxylates:
                        for carboxylate2 in new_entity.carboxylates:
                            if carboxylate1.carboxylates_superimposed(carboxylate2):
                                pair = (carboxylate1, carboxylate2)
                                additional_to_be_connect_carboxylate_pairs.append(pair)
                                additional_to_be_connect_carboxylate_on_new_entity.append(carboxylate2)
                                Is_bumping = False
                                break
                        if not Is_bumping:
                            break

                # Check for bumping if no connections are found
                if Is_bumping:
                    Is_bumping = self.kdtree_method(new_entity, entity)
                if Is_bumping:
                    self.time_for_judge_bumping += (time.time()-start_time)
                    return Is_bumping, additional_to_be_connect_carboxylate_pairs, additional_to_be_connect_carboxylate_on_new_entity

        self.time_for_judge_bumping += (time.time()-start_time)
        return Is_bumping, additional_to_be_connect_carboxylate_pairs, additional_to_be_connect_carboxylate_on_new_entity
    
    def add_entity(self, selected_carboxylate, new_entity, to_be_link_carboxylate_on_new_entity, additional_to_be_connect_carboxylate_pairs, additional_to_be_connect_carboxylate_on_new_entity):
        """
        Add a new entity (Zr6 or BTB) to the assembly and update connections.
        :param selected_carboxylate: The carboxylate to link with the new entity.
        :param new_entity: The new entity being added.
        :param to_be_link_carboxylate_on_new_entity: The carboxylate on the new entity to link.
        :param additional_to_be_connect_carboxylate_pairs: Additional carboxylate pairs to be connected.
        """
        # Add the new entity to the assembly
        # self.entities.append(new_entity)
        self.entities.add(new_entity)  ### Modified

        # Update the node mapping
        cube_index = tuple(np.floor(new_entity.center/self.ROI_range).astype(int))  # The cube index in which the center of the entity is located
        if cube_index not in self.node_mapping:
            self.node_mapping[cube_index] = [new_entity]
        else:
            self.node_mapping[cube_index].append(new_entity)

        # Update connections
        new_entity.connected_entities.append(selected_carboxylate.belonging_entity)
        selected_carboxylate.belonging_entity.connected_entities.append(new_entity)

        # Handle additional connections
        for pair in additional_to_be_connect_carboxylate_pairs:
            # self.ready_to_connect_carboxylate_pairs.append(pair)
            self.ready_to_connect_carboxylate_pairs.add(pair)
            for c in pair:
                self.pair_index[c] = pair
                if c in self.free_carboxylates:                                         
                    self.free_carboxylates.remove(c)                                    
                    if c.belonging_entity.entity_type == 'Zr':
                        self.MC_free_carboxylates.remove(c)
                    else:
                        self.Linker_free_carboxylates.remove(c)     
        # Update linked and free carboxylates
        pair = (selected_carboxylate, to_be_link_carboxylate_on_new_entity)       
        # self.linked_carboxylate_pairs.append((selected_carboxylate, to_be_link_carboxylate_on_new_entity))
        self.linked_carboxylate_pairs.add(pair)  ### Modified
        for c in pair:
            self.pair_index[c] = pair 
        self.free_carboxylates.remove(selected_carboxylate)
        if selected_carboxylate.belonging_entity.entity_type == 'Zr':
            self.MC_free_carboxylates.remove(selected_carboxylate)
        else:
            self.Linker_free_carboxylates.remove(selected_carboxylate)     
        # Add unlinked carboxylates from the new entity to free carboxylates        
        # self.free_carboxylates.extend([
        #     carboxylate for carboxylate in new_entity.carboxylates
        #     if (carboxylate not in additional_to_be_connect_carboxylate_on_new_entity) and (carboxylate != to_be_link_carboxylate_on_new_entity)
        #     ])
        for carboxylate in new_entity.carboxylates:                                      
            if carboxylate not in additional_to_be_connect_carboxylate_on_new_entity and carboxylate != to_be_link_carboxylate_on_new_entity:
                self.free_carboxylates.add(carboxylate)
                if carboxylate.belonging_entity.entity_type == 'Zr':
                    self.MC_free_carboxylates.add(carboxylate)
                else:
                    self.Linker_free_carboxylates.add(carboxylate)

    def remove_linkage(self, selected_pair):
        """
        Remove an entity from the assembly.
        """
        start_time = time.time()
        to_remove_entity = []
        to_search_entity = []

        if len(self.entities) > 2:
            if len(selected_pair[0].belonging_entity.connected_entities) == 1:
                to_remove_entity.append(selected_pair[0].belonging_entity)
            elif len(selected_pair[1].belonging_entity.connected_entities) == 1:
                to_remove_entity.append(selected_pair[1].belonging_entity)
            else:
                to_search_entity = [selected_pair[0].belonging_entity, selected_pair[1].belonging_entity]  
        else:
            to_remove_entity.append(selected_pair[0].belonging_entity)

        # print('entity to remove:',to_remove_entity)

        # Update carboxylate connections

        # Update linked pairs and carboxylates
        if selected_pair in self.linked_carboxylate_pairs:
            self.linked_carboxylate_pairs.remove(selected_pair)                                 
        else:
            self.linked_carboxylate_pairs.remove((selected_pair[1], selected_pair[0]))   
        # Update ready-to-connect pairs and carboxylates
        # self.ready_to_connect_carboxylate_pairs.append(selected_pair)
        self.ready_to_connect_carboxylate_pairs.add(selected_pair)
       
        # Update connection properties
        selected_pair[1].belonging_entity.connected_entities.remove(selected_pair[0].belonging_entity)
        selected_pair[0].belonging_entity.connected_entities.remove(selected_pair[1].belonging_entity)
        # del selected_pair

        #************************************************TEST*******************************************

        # print('to_search_entity:',to_search_entity)
        # for entity in self.entities:
        #     print(f"{entity}:{entity.connected_entities}")


        if len(to_search_entity) > 0:

            # 初始化双向搜索
            early_stop = False
            max_depth = MAX_DEPTH

            # 从两个实体同时开始搜索
            entity1, entity2 = to_search_entity
            queue1 = deque([entity1])
            queue2 = deque([entity2])
            visited1 = {entity1}
            visited2 = {entity2}

            depth1 = 0
            depth2 = 0
            level_size1 = len(queue1)
            level_size2 = len(queue2)
            
            # 记录每一层的增长情况
            prev_size1 = 1
            prev_size2 = 1
            no_growth1 = False
            no_growth2 = False
            
            # 双向BFS搜索
            while depth1 < max_depth and depth2 < max_depth:
                # 从第一个方向搜索
                if not early_stop and depth1 < max_depth:
                    entity = queue1.popleft()
                    level_size1 -= 1
                    
                    for connected_entity in entity.connected_entities:
                        # 检查是否与另一个搜索方向相遇
                        if connected_entity in visited2:
                            # 两个组件连通，不需要删除
                            early_stop = True
                            break
                        elif connected_entity not in visited1:
                            visited1.add(connected_entity)
                            queue1.append(connected_entity)
                    
                    # 当前层级处理完毕
                    if level_size1 == 0 and not early_stop:
                        depth1 += 1
                        level_size1 = len(queue1)
                        # 检查是否有增长
                        if len(visited1) == prev_size1:
                            no_growth1 = True
                            early_stop = True
                        prev_size1 = len(visited1)
                
                # 从第二个方向搜索
                if not early_stop and depth2 < max_depth:
                    entity = queue2.popleft()
                    level_size2 -= 1
                    
                    for connected_entity in entity.connected_entities:
                        # 检查是否与另一个搜索方向相遇
                        if connected_entity in visited1:
                            # 两个组件连通，不需要删除
                            early_stop = True
                            break
                        if connected_entity not in visited2:
                            visited2.add(connected_entity)
                            queue2.append(connected_entity)
                    
                    # 当前层级处理完毕
                    if level_size2 == 0 and not early_stop:
                        depth2 += 1
                        level_size2 = len(queue2)
                        # 检查是否有增长
                        if len(visited2) == prev_size2:
                            no_growth2 = True
                            early_stop = True
                        prev_size2 = len(visited2)
                
                if early_stop:
                    break
            
            # 如果搜索完成后某一边不再增长
            if no_growth1:
                # 如果第一组不再增长，判断其大小
                if len(visited1) > len(self.entities)/2:
                    to_remove_entity = list(set(self.entities) - visited1)
                else:
                    to_remove_entity = list(visited1)

            elif no_growth2:
                # 如果第二组不再增长，判断其大小
                if len(visited2) > len(self.entities)/2:
                    to_remove_entity = list(set(self.entities) - visited2)
                else:
                    to_remove_entity = list(visited2)
        
        # if len(to_remove_entity) > 1:
        #     print(f"removing {len(to_remove_entity)} entities from {len(self.entities)}")


        #************************************************TEST*******************************************

        end_time = time.time()
        self.time_for_cut_a_bond += (end_time-start_time)

        start_time = time.time()
        for entity in to_remove_entity:
            for carboxylate1 in entity.carboxylates:
                if carboxylate1 in self.free_carboxylates:
                    self.free_carboxylates.remove(carboxylate1)                    
                    if carboxylate1.belonging_entity.entity_type == 'Zr':
                        self.MC_free_carboxylates.remove(carboxylate1)
                    else:
                        self.Linker_free_carboxylates.remove(carboxylate1) 
                else:
                    # pair = next((p for p in self.ready_to_connect_carboxylate_pairs if carboxylate1 in p), None)    
                    pair = self.pair_index.get(carboxylate1,None)  ### Modified
                    if pair:
                        the_other_carboxylate = pair[0] if carboxylate1 == pair[1] else pair[1]
                        if pair in self.ready_to_connect_carboxylate_pairs:
                            self.ready_to_connect_carboxylate_pairs.remove(pair)
                        elif pair in self.linked_carboxylate_pairs:
                            self.linked_carboxylate_pairs.remove(pair)
                        for c in pair:
                            self.pair_index.pop(c,None)
                        del pair                                        
                        # self.free_carboxylates.append(the_other_carboxylate)
                        self.free_carboxylates.add(the_other_carboxylate)  ### Modified
                        
                        if the_other_carboxylate.belonging_entity.entity_type == 'Zr':
                            self.MC_free_carboxylates.add(the_other_carboxylate)
                        else:
                            self.Linker_free_carboxylates.add(the_other_carboxylate)
                carboxylate1.belonging_entity = None

            entity.kdtree = None

            for connected_entity in entity.connected_entities:
                connected_entity.connected_entities.remove(entity)
            entity.connected_entities = None
                                                                          
            # Remove the entity from node_mapping
            cube_index = tuple(np.floor(entity.center/self.ROI_range).astype(int))
            self.node_mapping[cube_index].remove(entity)
            if len(self.node_mapping[cube_index]) == 0:
                del self.node_mapping[cube_index]

            # Remove the entity from the assembly    
            self.entities.remove(entity)  
            # to_remove_entity.remove(entity)
            # del entity
        # if len(to_remove_entity) > 1:
        #     print(f"after removing, there are {len(self.entities)} entities left")
        to_remove_entity = []
        self.time_for_remove_entities += (time.time()-start_time)

    def visualize(self):
        """
        Visualize the assembly.
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = {'Zr': 'b', 'Ligand': 'r', 'Carboxylate':'y'}

        for entity in self.entities:
            xs, ys, zs = zip(*[coord for coord in entity.coordinates])
            ax.scatter(xs, ys, zs, c=colors[entity.entity_type], marker='o')
            
        for carboxylate in self.free_carboxylates:
            xs, ys, zs = zip(*[coord for coord in carboxylate.belonging_entity.coordinates[carboxylate.atom_indexes]])
            ax.scatter(xs, ys, zs, c=colors['Carboxylate'], marker='o')
            
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.show()

    
    def get_atom_list(self):
        atom_lists = []
        for entity in self.entities:
            for i in range(len(entity.elements)):
                atom_lists.append([entity.elements[i],entity.coordinates[i][0],entity.coordinates[i][1],entity.coordinates[i][2]])
        return atom_lists
    
    def get_mol_file(self, filename):
        atom_list = self.get_atom_list()
            
        num_atoms =len(atom_list)
    
        with open(filename, 'w') as f:
            # Write the header
            f.write("GeneratedMol\n")
            f.write("  Assembly            3D\n")
            f.write("\n")
            
            # Write the counts line
            f.write(f"{num_atoms:3d}  0  0  0  0  0  0  0  0  0  0999 V2000\n")
            
            # Write the atom block
            for atom in atom_list:
                f.write(f"   {atom[1]:10.4f}   {atom[2]:10.4f}   {atom[3]:10.4f} {atom[0]:2s}  0  0  0  0  0  0  0  0  0  0\n")
            
            # Write the footer
            f.write("M  END\n")

    def get_mol2_file(self,filename):
        atom_list = self.get_atom_list()      
        num_atoms =len(atom_list)
        num_bonds = 0  # Assuming no bond information in numpy array; this can be adjusted if bond information is provided separately
    
        with open(filename, 'w') as f:
            # Write the header
            f.write("# Mol2 file generated by python\n")
            f.write("@<TRIPOS>MOLECULE\n")
            f.write('"Generated Molecule"\n')
            f.write(f"{num_atoms} {num_bonds} 0 0 0\n")
            f.write("SMALL\n")
            f.write("USER_CHARGES\n\n")
            
            # Write the atom block
            f.write("@<TRIPOS>ATOM\n")
            for i, atom in enumerate(atom_list, start=1):
                f.write(f"{i} {atom[0]} {atom[1]:.5f} {atom[2]:.5f} {atom[3]:.5f} {atom[0]} 0 **** 0\n")
            
            # Write the bond block
            f.write("@<TRIPOS>BOND\n")
            # Note: Bonds need to be defined here. If not available, this section will remain without bonds.
            # Example: f.write("1 1 2 1\n")
            
            # Closing the file
            f.write("\n")

# --------------------------Tool Function--------------------------------
import pickle
import sys
from pathlib import Path

def thorough_cleanup_for_pickle(assembly, clean_connected_entities=False):
    """
    Thorough cleanup before save: Remove all circular references that could cause recursion
    
    Parameters:
    - assembly: Assembly object
    - clean_connected_entities: Whether to clean connected_entities (default False, preserves connection info)
    
    Note: This function modifies the assembly object. Call rebuild_references_after_load after saving to restore.
    """
    print("=" * 60)
    print("Starting thorough cleanup for pickle...")
    print("=" * 60)
    
    cleanup_stats = {
        'kdtrees_cleaned': 0,
        'carboxylates_cleaned': 0,
        'entities_cleaned': 0,
        'connected_entities_cleaned': 0,
    }
    
    # ========== Step 1: Clean KDTree ==========
    print("\n[1/4] Cleaning KDTree...")
    if hasattr(assembly, 'entities') and assembly.entities:
        for entity in assembly.entities:
            if entity is not None:
                if hasattr(entity, 'kdtree') and entity.kdtree is not None:
                    entity.kdtree = None
                    cleanup_stats['kdtrees_cleaned'] += 1
                
                # Check kdtree in nested objects
                for attr_name in dir(entity):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(entity, attr_name)
                            if hasattr(attr_value, 'kdtree'):
                                attr_value.kdtree = None
                        except:
                            pass
    
    print(f"   Cleaned {cleanup_stats['kdtrees_cleaned']} KDTree objects")
    
    # ========== Step 2: Collect all Carboxylate objects ==========
    print("\n[2/4] Collecting all Carboxylate objects...")
    all_carboxylates = set()
    
    # Collect from entities
    if hasattr(assembly, 'entities') and assembly.entities:
        for entity in assembly.entities:
            if entity is not None and hasattr(entity, 'carboxylates'):
                if entity.carboxylates:
                    for carb in entity.carboxylates:
                        if carb is not None:
                            all_carboxylates.add(carb)
    
    # Collect from various carboxylate sets
    carb_set_names = [
        'free_carboxylates',
        'MC_free_carboxylates',
        'Linker_free_carboxylates'
    ]
    
    for carb_set_name in carb_set_names:
        if hasattr(assembly, carb_set_name):
            carb_set = getattr(assembly, carb_set_name)
            if carb_set:
                for carb in carb_set:
                    if carb is not None:
                        all_carboxylates.add(carb)
    
    # Collect from pair sets
    pair_set_names = [
        'linked_carboxylate_pairs',
        'ready_to_connect_carboxylate_pairs'
    ]
    
    for pair_set_name in pair_set_names:
        if hasattr(assembly, pair_set_name):
            pair_set = getattr(assembly, pair_set_name)
            if pair_set:
                for pair in pair_set:
                    if pair:
                        for carb in pair:
                            if carb is not None:
                                all_carboxylates.add(carb)
    
    # Collect from pair_index
    if hasattr(assembly, 'pair_index') and assembly.pair_index:
        for carb, pair in assembly.pair_index.items():
            if carb is not None:
                all_carboxylates.add(carb)
            if pair:
                for c in pair:
                    if c is not None:
                        all_carboxylates.add(c)
    
    print(f"   Found {len(all_carboxylates)} Carboxylate objects")
    
    # ========== Step 3: Clean Carboxylate.belonging_entity (Core!) ==========
    print("\n[3/4] Cleaning Carboxylate.belonging_entity (breaking circular references)...")
    
    for carb in all_carboxylates:
        if carb is not None:
            # Clean belonging_entity reference
            if hasattr(carb, 'belonging_entity'):
                if carb.belonging_entity is not None:
                    carb.belonging_entity = None
                    cleanup_stats['carboxylates_cleaned'] += 1
            
            # Clean possible kdtree
            if hasattr(carb, 'kdtree'):
                carb.kdtree = None
    
    print(f"   Cleaned belonging_entity for {cleanup_stats['carboxylates_cleaned']} Carboxylates")
    
    # ========== Step 4: Clean Entity.connected_entities (Optional) ==========
    print("\n[4/4] Processing Entity.connected_entities...")
    
    if clean_connected_entities:
        print("   Cleaning connected_entities (will lose connection topology, use with caution)...")
        if hasattr(assembly, 'entities') and assembly.entities:
            for entity in assembly.entities:
                if entity is not None:
                    if hasattr(entity, 'connected_entities') and entity.connected_entities:
                        entity.connected_entities = []
                        cleanup_stats['connected_entities_cleaned'] += 1
        print(f"   Cleaned connected_entities for {cleanup_stats['connected_entities_cleaned']} Entities")
    else:
        print("   Keeping connected_entities (recommended, preserves connection topology)")
        print("   If RecursionError persists, set clean_connected_entities=True")
    
    # ========== Cleanup Complete ==========
    print("\n" + "=" * 60)
    print("Cleanup complete! Statistics:")
    print(f"  - KDTree cleaned: {cleanup_stats['kdtrees_cleaned']}")
    print(f"  - Carboxylate cleaned: {cleanup_stats['carboxylates_cleaned']}")
    print(f"  - Connected Entities cleaned: {cleanup_stats['connected_entities_cleaned']}")
    print("=" * 60)
    
    return cleanup_stats


def rebuild_references_after_load(assembly, verbose=True):
    """
    Rebuild after load: Restore all cleaned references
    
    Parameters:
    - assembly: Loaded Assembly object
    - verbose: Whether to print detailed information
    
    Note: Must be called immediately after pickle.load, or before continuing execution after cleanup
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Starting reference rebuild...")
        print("=" * 60)
    
    rebuild_stats = {
        'entities_processed': 0,
        'carboxylates_rebuilt': 0,
        'connected_pairs_rebuilt': 0,
        'kdtrees_available': 0,
    }
    
    # ========== Step 1: Build Carboxylate -> Entity mapping ==========
    if verbose:
        print("\n[1/4] Building Carboxylate -> Entity mapping...")
    
    carb_to_entity = {}  # Mapping table: carboxylate -> entity
    
    if hasattr(assembly, 'entities') and assembly.entities:
        for entity in assembly.entities:
            if entity is not None:
                rebuild_stats['entities_processed'] += 1
                
                if hasattr(entity, 'carboxylates') and entity.carboxylates:
                    for carb in entity.carboxylates:
                        if carb is not None:
                            # Build mapping relationship
                            carb_to_entity[id(carb)] = entity
    
    if verbose:
        print(f"   Built mapping for {len(carb_to_entity)} Carboxylates")
        print(f"   Processed {rebuild_stats['entities_processed']} Entities")
    
    # ========== Step 2: Collect and rebuild all Carboxylate.belonging_entity ==========
    if verbose:
        print("\n[2/4] Rebuilding all Carboxylate.belonging_entity...")
    
    all_carboxylates = set()
    
    # Collect from entities
    if hasattr(assembly, 'entities') and assembly.entities:
        for entity in assembly.entities:
            if entity is not None and hasattr(entity, 'carboxylates'):
                if entity.carboxylates:
                    for carb in entity.carboxylates:
                        if carb is not None:
                            all_carboxylates.add(carb)
    
    # Collect from various carboxylate sets
    carb_set_names = [
        'free_carboxylates',
        'MC_free_carboxylates',
        'Linker_free_carboxylates'
    ]
    
    for carb_set_name in carb_set_names:
        if hasattr(assembly, carb_set_name):
            carb_set = getattr(assembly, carb_set_name)
            if carb_set:
                for carb in carb_set:
                    if carb is not None:
                        all_carboxylates.add(carb)
    
    # Collect from pair sets
    pair_set_names = [
        'linked_carboxylate_pairs',
        'ready_to_connect_carboxylate_pairs'
    ]
    
    for pair_set_name in pair_set_names:
        if hasattr(assembly, pair_set_name):
            pair_set = getattr(assembly, pair_set_name)
            if pair_set:
                for pair in pair_set:
                    if pair:
                        for carb in pair:
                            if carb is not None:
                                all_carboxylates.add(carb)
    
    # Collect from pair_index
    if hasattr(assembly, 'pair_index') and assembly.pair_index:
        for carb, pair in assembly.pair_index.items():
            if carb is not None:
                all_carboxylates.add(carb)
            if pair:
                for c in pair:
                    if c is not None:
                        all_carboxylates.add(c)
    
    # Rebuild belonging_entity for all carboxylates
    unrebuilt_carboxylates = []
    for carb in all_carboxylates:
        if carb is not None:
            carb_id = id(carb)
            if carb_id in carb_to_entity:
                carb.belonging_entity = carb_to_entity[carb_id]
                rebuild_stats['carboxylates_rebuilt'] += 1
            else:
                unrebuilt_carboxylates.append(carb)
    
    # Report any carboxylates that couldn't be rebuilt
    if unrebuilt_carboxylates and verbose:
        print(f"   Warning: {len(unrebuilt_carboxylates)} carboxylates could not be mapped to entities")
        print(f"   These carboxylates may be orphaned or belong to removed entities")
    
    if verbose:
        print(f"   Found {len(all_carboxylates)} Carboxylate objects")
        print(f"   Rebuilt belonging_entity for {rebuild_stats['carboxylates_rebuilt']} Carboxylates")
    
    # ========== Step 3: Rebuild Entity.connected_entities ==========
    if verbose:
        print("\n[3/4] Rebuilding Entity.connected_entities...")
    
    # First clear all connected_entities
    if hasattr(assembly, 'entities') and assembly.entities:
        for entity in assembly.entities:
            if entity is not None:
                if hasattr(entity, 'connected_entities'):
                    entity.connected_entities = []
    
    # Rebuild connection relationships from linked_carboxylate_pairs
    connected_pairs_count = 0
    if hasattr(assembly, 'linked_carboxylate_pairs') and assembly.linked_carboxylate_pairs:
        for pair in assembly.linked_carboxylate_pairs:
            if pair and len(pair) == 2:
                carb1, carb2 = pair
                if carb1 is not None and carb2 is not None:
                    entity1 = carb1.belonging_entity
                    entity2 = carb2.belonging_entity
                    
                    if entity1 is not None and entity2 is not None:
                        # Avoid duplicate additions
                        if entity2 not in entity1.connected_entities:
                            entity1.connected_entities.append(entity2)
                        if entity1 not in entity2.connected_entities:
                            entity2.connected_entities.append(entity1)
                        connected_pairs_count += 1
    
    if verbose:
        print(f"   Rebuilt connected_entities from {connected_pairs_count} connection pairs")
        rebuild_stats['connected_pairs_rebuilt'] = connected_pairs_count
    
    # ========== Step 4: Check KDTree status (not auto-rebuilt, rebuilt on demand) ==========
    if verbose:
        print("\n[4/4] Checking KDTree status...")
    
    if hasattr(assembly, 'entities') and assembly.entities:
        for entity in assembly.entities:
            if entity is not None:
                if hasattr(entity, 'kdtree') and entity.kdtree is not None:
                    rebuild_stats['kdtrees_available'] += 1
    
    if verbose:
        if rebuild_stats['kdtrees_available'] > 0:
            print(f"    Found {rebuild_stats['kdtrees_available']} Entities with existing KDTree")
        else:
            print(f"   All KDTrees cleared (will be rebuilt on demand)")
        print("    KDTrees will be automatically rebuilt when used (lazy loading)")
    
    # ========== Verification: Check reference integrity ==========
    if verbose:
        print("\n[Verification] Checking reference integrity...")
        
        # Verify carboxylate references
        broken_refs = 0
        total_checked = 0
        
        # Check all collected carboxylates
        for carb in all_carboxylates:
            if carb is not None and hasattr(carb, 'belonging_entity'):
                total_checked += 1
                if carb.belonging_entity is None:
                    broken_refs += 1
        
        # Check ready_to_connect_carboxylate_pairs specifically
        ready_broken = 0
        if hasattr(assembly, 'ready_to_connect_carboxylate_pairs') and assembly.ready_to_connect_carboxylate_pairs:
            for pair in assembly.ready_to_connect_carboxylate_pairs:
                if pair and len(pair) == 2:
                    c1, c2 = pair
                    if (not hasattr(c1, 'belonging_entity') or c1.belonging_entity is None or
                        not hasattr(c2, 'belonging_entity') or c2.belonging_entity is None):
                        ready_broken += 1
        
        if broken_refs > 0:
            print(f"   Warning: Found {broken_refs}/{total_checked} Carboxylates with None belonging_entity")
            print(f"   These Carboxylates may not belong to any Entity")
        else:
            print(f"   All {total_checked} Carboxylate belonging_entity references are valid")
        
        if ready_broken > 0:
            print(f"   Warning: Found {ready_broken} ready pairs with None belonging_entity")
            print(f"   Cleaning up invalid ready pairs...")
            # Clean up invalid ready pairs
            if hasattr(assembly, 'ready_to_connect_carboxylate_pairs') and assembly.ready_to_connect_carboxylate_pairs:
                valid_ready_pairs = []
                for pair in assembly.ready_to_connect_carboxylate_pairs:
                    if pair and len(pair) == 2:
                        c1, c2 = pair
                        if (hasattr(c1, 'belonging_entity') and c1.belonging_entity is not None and
                            hasattr(c2, 'belonging_entity') and c2.belonging_entity is not None):
                            valid_ready_pairs.append(pair)
                # Update the ready_to_connect_carboxylate_pairs
                if hasattr(assembly.ready_to_connect_carboxylate_pairs, 'update'):
                    # RandomizedSet or set
                    assembly.ready_to_connect_carboxylate_pairs = type(assembly.ready_to_connect_carboxylate_pairs)()
                    assembly.ready_to_connect_carboxylate_pairs.update(valid_ready_pairs)
                else:
                    # Regular list
                    assembly.ready_to_connect_carboxylate_pairs = valid_ready_pairs
                
                # Clean up pair_index: remove entries for invalid pairs
                if hasattr(assembly, 'pair_index') and assembly.pair_index:
                    valid_pairs_set = set(valid_ready_pairs)
                    keys_to_remove = []
                    for carb, pair in assembly.pair_index.items():
                        if pair not in valid_pairs_set and pair not in getattr(assembly, 'linked_carboxylate_pairs', set()):
                            keys_to_remove.append(carb)
                    for key in keys_to_remove:
                        assembly.pair_index.pop(key, None)
                
                print(f"   Removed {ready_broken} invalid ready pairs, {len(valid_ready_pairs)} remain")
    
    # ========== Rebuild Complete ==========
    if verbose:
        print("\n" + "=" * 60)
        print("Rebuild complete! Statistics:")
        print(f"  - Entities processed: {rebuild_stats['entities_processed']}")
        print(f"  - Carboxylate references rebuilt: {rebuild_stats['carboxylates_rebuilt']}")
        print(f"  - Connection relationships rebuilt: {rebuild_stats['connected_pairs_rebuilt']} pairs")
        print(f"  - Existing KDTrees: {rebuild_stats['kdtrees_available']}")
        print("=" * 60)
    
    return rebuild_stats


# ============================================================
# Convenience functions: Safe save and load
# ============================================================

def safe_pickle_save(assembly, filepath, clean_connected_entities=False, 
                     rebuild_after_save=True, protocol=None):
    """
    Safely save Assembly object to pickle file
    
    Parameters:
    - assembly: Assembly object to save
    - filepath: Save path
    - clean_connected_entities: Whether to clean connected_entities
    - rebuild_after_save: Whether to immediately rebuild references after saving (if continuing execution)
    - protocol: Pickle protocol version (defaults to highest version)
    
    Returns:
    - success: Whether successful
    """
    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL
    
    print(f"\nSafely saving Assembly to: {filepath}")
    
    try:
        # 1. Cleanup
        thorough_cleanup_for_pickle(assembly, clean_connected_entities=clean_connected_entities)
        
        # 2. Save
        print(f"\nWriting pickle file...")
        with open(filepath, 'wb') as f:
            pickle.dump(assembly, f, protocol=protocol)
        print(f"Successfully saved to: {filepath}")
        
        # 3. Rebuild (if continuing execution)
        if rebuild_after_save:
            print(f"\nRebuilding references after save (continue execution mode)...")
            rebuild_references_after_load(assembly, verbose=True)
        
        return True
        
    except RecursionError as e:
        print(f"\n RecursionError: Recursion error persists even after cleanup!")
        print(f"   Error message: {str(e)}")
        print(f"   Suggestion: Try setting clean_connected_entities=True")
        return False
        
    except Exception as e:
        print(f"\n Save failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def safe_pickle_load(filepath, rebuild_references=True):
    """
    Safely load Assembly object from pickle file
    
    Parameters:
    - filepath: Pickle file path
    - rebuild_references: Whether to rebuild references
    
    Returns:
    - assembly: Loaded Assembly object, returns None on failure
    """
    print(f"\nLoading from pickle file: {filepath}")
    
    try:
        # 1. Load
        with open(filepath, 'rb') as f:
            assembly = pickle.load(f)
        print(f"Successfully loaded pickle file")
        
        # 2. Rebuild
        if rebuild_references:
            rebuild_references_after_load(assembly, verbose=True)
        
        return assembly
        
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
        
    except Exception as e:
        print(f"Load failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
