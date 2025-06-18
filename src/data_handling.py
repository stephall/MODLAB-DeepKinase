# data_handling.py

# Import public modules
import random
import torch
import torch_geometric
import numpy as np

# Import custom modules
from . import random_handling
from . import data_preprocessing

# Define a class for lists of torch data objects
class ListDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, data_list, device=torch.device('cpu')):
        # Initialize the base class without anything
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        
        # Collate the passed data_list and assign the output to the 'data' and 'slices' class attributes
        data, self.slices = self.collate(data_list)

        # Send the data to the device
        self.data = data.to(device)


# Define the DataHandler class
class DataHandler(object):
    def __init__(self, 
                 data_handling_config_dict, 
                 data_preprocessing_logger=None):
        """
        Initialize a data handler that (i) will preprocess the data during initialization
        and (ii) can be (after initialization) used for any actions (such as creating
        torch dataloader objects for model training or evalation) involving the data.

        Args:
            data_handling_config_dict (dict): Config dictionary for data handling that must 
                contain key-value pairs:
                'batch_size' (int): Batch size for the dataloaders.
                    (Default: 128)
                'num_workers' (int): Number of workers (extra sub processed in addition to main process) used by torch dataloaders.
                    Remark: If num_workers of a torch dataloader is 0, the main process loads the data.
                            If 0<num_workers of a torch dataloader, sub processes (in addition to the main process) are launched to load the data.
                    (Default: 0)
                'random_seed_train_dataloader' (int): Random seed that will be used by the train-dataloader to provide batches
                    (whose order should be randomized for each epoch)
                    (Default: 43)
            data_preprocessing_logger (logging.logger or None): Logger for all the data preprocessing or None.
                (Default: None)
        
        """
        # Assign entries in the data handling config dictionary to class attributes and define
        # default values to be used in case the key-value pair is not contained.
        self.batch_size                   = data_handling_config_dict.get('batch_size', 128)
        self.num_workers                  = data_handling_config_dict.get('num_workers', 0)
        self.random_seed_train_dataloader = data_handling_config_dict.get('random_seed_train_dataloader', 43)
    
        # Initialize a random handler object and assign it to a class attribute
        self._random_handler = random_handling.RandomHandler()

        # Define the set names attribute
        self.set_names = ['train', 'valid', 'test']

        ##############################################################################################################################
        # Encapsulate randomness in data preprocessing
        ##############################################################################################################################
        # Save the external random states
        self._random_handler.save_states()

        # Instantiate the data preprocessor
        data_preprocessing_config_dict = data_handling_config_dict['data_preprocessing']
        self.data_preprocessor = data_preprocessing.DataPreprocessor(data_preprocessing_config_dict, data_preprocessing_logger)

        # Run the preprocessor
        self.data_preprocessor.run()

        # Reset the external random states after data preprocessing
        self._random_handler.reset_states()
        ##############################################################################################################################

    @property
    def dataset_df(self):
        """ Return the dataset DataFrame that corresponds to 'self.data_preprocessor.preprocessed_df'. """
        return self.data_preprocessor.preprocessed_df

    @property
    def num_datapoints(self):
        """ Return the number of data points contained in the dataset (corresponding to the number or entries/rows of the dataset DataFrame). """
        return len(self.dataset_df)

    def get_num_datapoints_set(self, 
                               set_name):
        """ 
        Return the number of data points in the specified set.

        Args:
            set_name (str): Name of the set (e.g., 'train', 'valid', or 'test').

        Return:
            (int): Number of datapoints in the set.
        
        """
        if set_name in self.data_preprocessor.set_to_processed_subset_df_map:
            return len(self.data_preprocessor.set_to_processed_subset_df_map[set_name])
        else:
            err_msg = f"No subset of the data with name '{set_name}' is available.\nPlease use one of the following set names: {list(self.data_preprocessor.set_to_processed_subset_df_map.keys())}"
            raise ValueError(err_msg)

    def get_example_batch(self, 
                          set_name='train', 
                          device=None):
        """
        Return an example batch. 
        
        Args:
            set_name (str): Name of the set (e.g., 'train', 'valid', or 'test').
            device (torch.device or None): Device the datasets (for each train, valid, test) should be moved to or None.
                If the device is None, it will be selected within the method 'get_dataloader_dict'.
                (Default: None)

        Return:
            (torch.DataBatch): First batch of the specified (dataloader) set and fold. 
        
        """
        # Get the dataloader dictionary
        dataloader_dict = self.get_dataloader_dict(device)

        # Return the first element of the dataloader of the specified set (via set_name)
        return next( iter(dataloader_dict[set_name]) )

    def get_dataloader_dict(self, 
                            device=None):
        """
        Return a dictionary containing dataloader objects for the train, validation, and test subsets.

        Args:
            device (torch.device or None): Device the datasets (for each train, valid, test) should be moved to
                or None (in which case the device will be CUDA if it is available and otherwise CPU).
                (Default: None)
            
        Return:
            (dict): Dictionary containing three (pytorch or pytorch_geometric) dataloaders as values to the keys
                'train', 'valid', and 'test'.

        """
        # Check that the torch data lists of the different subsets has been defined in the data preprocessor
        if self.data_preprocessor.set_to_torch_data_list_map is None:
            err_msg = f"The attribute 'set_to_torch_data_list_map' has not been defined for the data preprocessor."
            raise AttributeError(err_msg)

        # Check that the keys of the this map correpond to the required set names
        if set(self.set_names)!=set(self.data_preprocessor.set_to_torch_data_list_map.keys()):
            err_msg = f"The keys of 'self.data_preprocessor.set_to_torch_data_list_map' should correspond to {self.set_names}, got the keys {list(self.data_preprocessor.set_to_torch_data_list_map.keys())} instead."
            raise KeyError(err_msg)

        # In case that the device is not passed (is None), use CUDA if available and CPU otherwise
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Differ cases for zero/one or multiple workers
        # Remarks: 1) If num_workers of a torch dataloader is 0, the main process loads the data.
        #             If 0<num_workers of a torch dataloader, sub processes (in addition to the main process) are launched to load the data.
        #          2) In case that the main process or only a single sub process should be used for data loading, there will only be one loading
        #             process and this process does not need an extra random seed. However, if 1<num_workers, the different sub processes should
        #             each have different random seeds and thus a function has to be defined that assigns each worker such a random seed.
        if self.num_workers==0 or self.num_workers==1:
            # Set the seed function of the loader workers to None
            seed_loader_worker_fn = None
        else:
            # Define seeds for the loader workers
            def seed_loader_worker_fn(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

        # Loop over the key-value pairs (set_name, set_torch_data_list) of the dictionary (in the data_preprocessor) that
        # contains the set names as dictionary-keys and the corresponding torch data lists as dictionary-values.
        # Create a torch dataloader for each set (train, valid, test)
        dataloader_dict = dict()
        for set_name, torch_data_list in self.data_preprocessor.set_to_torch_data_list_map.items():
            # In case that the current subset contains no data, assign None as dictionary-value for the set_name dictionary-key
            # and continue to the next subset
            # Remark: This can for example happen for the validation set in case that the model is retrained on the full 
            #         train-validation set where all of this data is assigned to the train and none is assigned to the 
            #         valididation set).
            if len(torch_data_list)==0:
                dataloader_dict[set_name] = None
                continue

            # Generate a ListDataset from the torch data list of the current set
            torch_dataset = ListDataset(torch_data_list, device=device)

            # Determine the appropriate DataLoader by differing the cases where the torch_dataset is a subclass
            # of 'torch_geometric.data.dataset.Dataset' (so graph structured data) or not
            if issubclass(type(torch_dataset), torch_geometric.data.dataset.Dataset):
                # In case the dataset is a subset of 'torch_geometric.data.dataset.Dataset', use the DataLoader of pytorch-geometric
                DataLoader = torch_geometric.loader.DataLoader
            else:
                # Otherwise, use the DataLoader of pytorch (non-geometric)
                DataLoader = torch.utils.data.DataLoader

            # Differ cases because the train dataloader should iterate randomly over batches, while the test and validation 
            # dataloaders do not need to do this.
            if set_name=='train':
                # During training, shuffle the batches for each epoch
                shuffle = True

                # Define a generator for the train dataloader
                # Remark: This will decouple any randomness in shuffling of the batches in the dataloader
                #         from any randomness in the model parameter intialization (or also potential sampling).
                generator = torch.Generator()
                generator.manual_seed(self.random_seed_train_dataloader)
            else:
                # During validation or testing, the batches do not need to be shuffeled for different epochs
                # because the whole datasets are usually iterated through without any changes to the model
                # so that the order of the batches doesn't matter
                shuffle = False

                # For the validation or testing dataloaders, we do not shuffle and thus do not need to
                # define a generator (for randomness) as the loading is purely determinisitc.
                generator = None

            # Define the dataloader for the current subset and assign it to the dataloader dictionary
            dataloader_dict[set_name] = DataLoader(torch_dataset, batch_size=self.batch_size,
                                                   shuffle=shuffle, generator=generator,
                                                   num_workers=self.num_workers, worker_init_fn=seed_loader_worker_fn)

        return dataloader_dict
