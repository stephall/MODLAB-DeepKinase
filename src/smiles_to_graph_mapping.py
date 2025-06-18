# smiles_to_graph_mapping.py

# Import public modules
import collections
import rdkit
import numpy as np
from rdkit import Chem

# Define a class to map SMILES strings to graphs
# Remark: Inspired by https://dmol.pub/dl/gnn.html
class SmilesToGraphMapper(object):
    # Define a dictionary of default vertex and edge feature types and their set of
    # possible values (as lists
    # For vertex features
    _default_vertex_features = {
        'atom_type':     ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'], # Further options: 'Si', 'Se', 'B'
        'hybridization': ['UNSPECIFIED', 'SP', 'SP2', 'SP3', 'S'], # Other options: 'SP3D', 'SP3D2'
        'is_in_ring':    [False, True],
        'is_aromatic':   [False, True],
    }

    # For edge features
    _default_edge_features = {
        'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    }

    # Define a map from vertex feature type a method of either 'rdkit.Chem.rdchem.Atom' or 'rdkit.Chem.rdchem.Bond'
    # used to obtain this feature from an atom (=vertex) or bond (=edge) RDKit object.
    _feature_type_to_rdkit_method_map = {
        'atom_type':     'GetSymbol',
        'hybridization': 'GetHybridization',
        'is_in_ring':    'IsInRing',
        'is_aromatic':   'GetIsAromatic',
        'bond_type':     'GetBondType', 
    }

    def __init__(self, 
                 add_Hs=True, 
                 use_edge_features=False, 
                 use_conformers=False, 
                 display_info=False, 
                 **kwargs):
        """
        Define a SMILES to Graph Mapper object.
        This object can be used to map a SMILES string, specifying a molecule, to a graph representation of
        the molecule where the atoms correspond to the vertices.

        Args:
            add_Hs (bool): Boolean flag if H-atoms should be added to the graph or not.
                (Default: True)
            use_edge_features (bool): Boolean flag if edge features should be included in the 
                graph or not. 
                (Default: False)
            use_conformers (bool): Boolean flag if conformers should be used for the molecules.
                (Default: False)
            display_info (bool): Boolean flag if additional information should be displayed.
                (Default: False)
            kwargs (dict): Key-word arguments that can be used to overwrite the set of values for 
                the different feature types.

        """
        # Assign inputs to class attributes
        self._add_Hs            = add_Hs
        self._use_edge_features = use_edge_features
        self._use_conformers    = use_conformers
        self._display_info      = display_info

        # Generate the vertex and edge feature maps
        self.vertex_feature_map = self._generate_vertex_features_map(kwargs)
        if self._use_edge_features:
            self.edge_feature_map = self._generate_edge_features_map(kwargs)
        else:
            self.edge_feature_map = dict()
        
    def _generate_vertex_features_map(self, 
                                      kwargs):
        """
        Creat a map (dictionary) from the vertex feature values to integers.
        Remark: Use passed vertex features instead of default ones if passed in the kwargs.

        Args:
            kwargs (dict): key-word arguments passed to __init__.

        Return:
            (dict): Vertex feature map.

        """
        # Initialize the vertex features map as empty dictionary and loop over all vertex feature types.
        vertex_features_map = dict()
        for vertex_feature_type in self._default_vertex_features:
            # Pop the vertex feature from the kwargs.
            # If it is not included in kwargs, return None
            vertex_feature_list = kwargs.pop(vertex_feature_type, None)
            
            # In case the vertex feature (list) was not passed as kwarg, use the default value
            if vertex_feature_list is None:
                vertex_feature_list = self._default_vertex_features[vertex_feature_type]

            # Create a map (dictionary) that maps a vertex feature value to an integer
            vertex_features_map[vertex_feature_type] = {key: value for value, key in enumerate(vertex_feature_list)}

        return vertex_features_map

    def _generate_edge_features_map(self, 
                                    kwargs):
        """
        Creat a map (dictionary) from the edge feature values to integers.
        Remark: Use passed edge features instead of default ones if passed in the kwargs.

        Args:
            kwargs (dict): key-word arguments passed to __init__.

        Return:
            (dict): Edge feature map.

        """
        # Initialize the edge features map as empty dictionary and loop over all edge feature types.
        edge_features_map = dict()
        for edge_feature_type in self._default_edge_features:
            # Pop the edge feature from the kwargs.
            # If it is not included in kwargs, return None
            edge_feature_list = kwargs.pop(edge_feature_type, None)
            
            # In case the edge feature (list) was not passed as kwarg, use the default value
            if edge_feature_list is None:
                edge_feature_list = self._default_edge_features[edge_feature_type]

            # Create a map (dictionary) that maps a edge feature value to an integer
            edge_features_map[edge_feature_type] = {key: value for value, key in enumerate(edge_feature_list)}

        return edge_features_map
    
    def _get_mol_obj(self, 
                     mol_smiles):
        """
        Return the molecule (rdkit.Chem.rdchem.Mol) object for an input smiles string.
        
        Args:
            mol_smiles (str): SMILES string of a molecule to be mapped to a molecule object.

        Returns:
            (rdkit.Chem.rdchem.Mol): Molecule object.
        
        """
        # Make a molecule (rdkit.Chem.rdchem.Mol) object from the SMILES string
        mol_obj = Chem.MolFromSmiles(mol_smiles)

        # Add H-atoms in case they should be added
        if self._add_Hs:
            mol_obj = Chem.AddHs(mol_obj)
            
        return mol_obj
    
    def _get_vertex_features_dict(self, 
                                  mol_obj):
        """
        Get the vertex features dictionary of a molecule.

        Args:
            mol_obj (rdkit.Chem.rdchem.Mol): Molecule as (RDKit) molecule object.

        Returns:
            (dict): Dictionary containing the vertex features (list) as values with the 
                vertex feature type (e.g. 'hybridization') as keys.
        """
        # Initialize the vertex features dictionary as defaults dictionary that will contain lists
        vertex_features_dict = collections.defaultdict(list)

        # Loop over the atoms (=vertices) in the graph
        for atom_obj in mol_obj.GetAtoms():
            # Loop over the vertex feature types
            for vertex_feature_type in self.vertex_feature_map:
                # Get a handle to the method of 'rdkit.Chem.rdchem.Atom' that will return the current value for 
                # the current vertex feature when called.
                method_handle = getattr(atom_obj, self._feature_type_to_rdkit_method_map[vertex_feature_type])
                vertex_feature_value = method_handle()

                # Cast the vertex feature value to a string (as it could be some RDKit data type)
                # unless it is a boolean
                if not isinstance(vertex_feature_value, bool):
                    vertex_feature_value = str(vertex_feature_value)
                
                # In case that the vertex feature value is not in the vertex feature map for the current
                # vertex feature type, inform the user and throw an error
                if vertex_feature_value not in self.vertex_feature_map[vertex_feature_type]:
                    if self._display_info:
                        print(f"The vertex feature value '{vertex_feature_value}' for type '{vertex_feature_type}' is not in the feature map that expects: {list(self.vertex_feature_map[vertex_feature_type].keys())}.")
                    raise MoleculeContainsNonFeaturizableAtom

                # Get the numeric (integer) representation of the current vertex feature and append it
                vertex_feature_numeric = self.vertex_feature_map[vertex_feature_type][vertex_feature_value]
                vertex_features_dict[vertex_feature_type].append(vertex_feature_numeric)

        return vertex_features_dict
    
    def _get_vertex_positions(self, mol_obj):
        """
        Get the vertex positions of a molecule. 
        """
        raise NotImplemented("The class method '_get_vertex_positions' has not been implemented.")
    
    def _get_edges_and_their_features(self, 
                                      mol_obj):
        """
        Get the edges (as 'edge_index' array) and edge features of a molecule.

        Args:
            mol_obj (rdkit.Chem.rdchem.Mol): Molecule as molecule object.

        Returns:
            (tuple(numpy.ndarray, dict)): A tuple containing an edge_index numpy array of shape 
                (2, #edges) and an edge_features_dict dictionary containing the edge feature types
                as keys and their corresponding 'edge feature values' as values.
        """
        # Check that the molecule has at least one bond (and must therefore be a set of unbonded atoms/ions)
        if len( mol_obj.GetBonds() )==0:
            err_msg = "The to be transformed molecule has no bonds and must therefore correspond to a set of unbonded atoms or ions.\n" \
                      "Defining a molecular graph to have at least one bond, the current molecule can not be transformed to a graph."
            raise MoleculeWithoutBondsError(err_msg)
        
        # Initialize an empty list and a defaults dictionary (for lists) for the edges and the edge features, respectively.
        edges              = list()
        edge_features_dict = collections.defaultdict(list)
        
        # Loop over the bonds
        for bond_obj in mol_obj.GetBonds():
            # Remark: A chemical bond corresponds to an undirected edge that can be represented by two
            #         directed edges with opposite directions [i.e. {A-B} = {A->B, A<-B} ]
            
            # Get the indices of the begin and end atom of the bond
            # Remark: Start and end atoms are defined by RDKit when constructing the molecular object.
            begin_atom_idx = bond_obj.GetBeginAtomIdx()
            end_atom_idx   = bond_obj.GetEndAtomIdx()
            
            # (1) Edges:
            # Append the two edges corresponding to the bond, which are both represented as pair of vertices, 
            # to the list of edges.
            # Append edge 'begin_atom_idx -> end_atom_idx' to the edges list
            edges.append([begin_atom_idx, end_atom_idx])
            
            # Append edge 'end_atom_idx -> begin_atom_idx' to the edges list
            edges.append([end_atom_idx, begin_atom_idx])
            
            # (2) Edge Features:
            # Loop over the edge feature types
            for edge_feature_type in self.edge_feature_map:
                # Get a handle to the method of 'rdkit.Chem.rdchem.Bond' that will return the current value for 
                # the current edge feature when called.
                method_handle = getattr(bond_obj, self._feature_type_to_rdkit_method_map[edge_feature_type])
                edge_feature_value = method_handle()

                # Cast the edge feature value to a string (as it could be some RDKit data type)
                # unless it is a boolean
                if not isinstance(edge_feature_value, bool):
                    edge_feature_value = str(edge_feature_value)

                # In case that the edge feature value is not in the edge feature map for the current
                # edge feature type, inform the user and throw an error
                if edge_feature_value not in self.edge_feature_map[edge_feature_type]:
                    if self._display_info:
                        print(f"The edge feature value '{edge_feature_value}' for type '{edge_feature_type}' is not in the feature map that expects: {list(self.edge_feature_map[edge_feature_type].keys())}.")
                    raise MoleculeContainsNonFeaturizableBond
                
                # Get the numeric (integer) representation of the current edge feature and append it
                edge_feature_numeric = self.edge_feature_map[edge_feature_type][edge_feature_value]
                edge_features_dict[edge_feature_type].append(edge_feature_numeric)
            
        # Create an edge_index numpy array that represents the adjacency matrix (and is usable in Pytorch_Geometric).
        edge_index = np.array(edges).T
        
        return edge_index, edge_features_dict
    
    def __call__(self, 
                 mol_smiles):
        """
        Map an input SMILES string to a graph specified by its vertex features and adjacency matrix.

        Args:
            mol_smiles (str): SMILES string of a molecule to be mapped to a graph.

        Returns:
            (dict): Graph dictionary containing the key-value pairs:
                - x_<vertex_feature_type> (numpy.ndarray) of shape (#vertices,) for each requested vertex feature types
                - vertex_positions (numpy.ndarray) of shape (#vertices, 3) [only determined if self._conformers=True]
                - edge_index (numpy.ndarray) of shape (2, #edges)
                - edge_attr_<edge_feature_type> (numpy.ndarray) of shape (#edges,) for each requested edge feature types
        """
        # Initialize a dictionary for the to be returned graph
        graph_dict = dict()
                             
        # Get the molecular object corresponding to the SMILES string.
        mol_obj = self._get_mol_obj(mol_smiles)

        # Get the number of atoms(=vertices) and add it as key-value pair
        graph_dict['num_vertices'] = len( mol_obj.GetAtoms() )
        
        # Try to get the vertex features dictionary
        try:
            vertex_features_dict = self._get_vertex_features_dict(mol_obj)
        except MoleculeContainsNonFeaturizableAtom:
            # If the molecule contains a non featurizable atom, it cannot be mapped to a graph.
            raise MoleculeCannotBeMappedToGraph

        # Loop over the vertex feature types and add them to the graph dictionary,
        # while mapping their values (the numeric features) from lists to numpy arrays
        for vertex_feature_type, vertex_features in vertex_features_dict.items():
            graph_dict[f"x_{vertex_feature_type}"] = np.array(vertex_features)
        
        # Get the vertex positions if conformers are used
        if self._use_conformers:
            graph_dict['vertex_positions'] = self._get_vertex_positions(mol_obj)
        
        # Try to get the edges (as edge_index) and their features
        try:       
            edge_index, edge_features_dict = self._get_edges_and_their_features(mol_obj)
        except MoleculeContainsNonFeaturizableBond:
            # If the molecule contains a non featurizable atom, it cannot be mapped to a graph.
            raise MoleculeCannotBeMappedToGraph
        
        # Assign the edge_index to the graph dictionary
        graph_dict['edge_index'] = edge_index

        # Loop over the edge feature types and add them to the graph dictionary,
        # while mapping their values (the numeric features) from lists to numpy arrays
        for edge_feature_type, edge_features in edge_features_dict.items():
            graph_dict[f"edge_attr_{edge_feature_type}"] = np.array(edge_features)

        # Return the graph dictionary
        return graph_dict

#############################################################################################################################
#############################################################################################################################
### Define auxiliary custom exceptions
#############################################################################################################################
#############################################################################################################################
class MoleculeWithoutBondsError(Exception):
    """ Define custom error that is raised, when a molecule without bonds should be transformed to a graph. """
    def __init__(self, message="Molecule has no bonds."):
        self.message = message
        super().__init__(self.message)

class MoleculeContainsNonFeaturizableAtom(Exception):
    """ Define custom error that is raised, when a molecule contains an atom that cannot be featurized. """
    def __init__(self, message="Molecule contains atom that cannot be featurized."):
        self.message = message
        super().__init__(self.message)

class MoleculeContainsNonFeaturizableBond(Exception):
    """ Define custom error that is raised, when a molecule contains a bond that cannot be featurized. """
    def __init__(self, message="Molecule contains bond that cannot be featurized."):
        self.message = message
        super().__init__(self.message)

class MoleculeCannotBeMappedToGraph(Exception):
    """ Define custom error that is raised, when a molecule cannot be mapped to a molecular graph because at least one of its atoms or bonds cannot be featurized. """
    def __init__(self, message="Molecule cannot be mapped to graph because it contains at least one atom or bond that cannot be featurized."):
        self.message = message
        super().__init__(self.message)