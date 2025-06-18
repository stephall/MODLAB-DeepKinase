# random_handling.py

# Import public modules
import random
import torch
import numpy as np

class RandomHandler(object):
    def __init__(self):
        # Initialize the attribute orig_states and seeds to None
        self.orig_states = None
        self.seeds       = None

    def set_seed(self, 
                 random_seed):
        """
        Set a random seed for the random-number-generators (rngs) of the random, numpy, 
        and pytorch (torch.random and torch.cuda) modules.
        Remark: One seed is used for of all available GPUs (so no individual seeds for
                any specific GPU is used).
            
        Args:
            random_seed (int): Non-negative integer random seed that will be used for as 
                seed for the random-number-generator (rng) of the random module. 
                For the numpy and torch modules a random seed 'random_seed+1' and 'random_seed+2',
                respectively, is used. 
        """
        # Check that random seed is a positive integer
        if not isinstance(random_seed, int):
            err_msg = f"The passed random seed must be an integer, got type {type(random_seed)} instead."
            raise TypeError(err_msg)

        if random_seed<0:
            err_msg = f"the passed random seed must be non-negative, got value {random_seed} instead."
            raise ValueError(err_msg)

        # Check that the attribute orig_states is None
        if self.orig_states is not None:
            err_msg = f"Can not set a random seed in case that the the original state has not already been reset (and thus self.orig_states is not None)."
            raise AttributeError(err_msg)

        # Get the current random states of the rngs of the different modules and save
        # them as the 'original states' so that they can be reset later.
        self.orig_states = dict()
        self.orig_states['random']         = random.getstate()
        self.orig_states['numpy']          = np.random.get_state()
        self.orig_states['torch_cuda_all'] = torch.cuda.get_rng_state_all()
        self.orig_states['torch_random']   = torch.random.get_rng_state()

        # Define the random seeds for the different modules based on the passed input
        # using the input for the random module, while adding 1 and 2 for the numpy and
        # torch (torch.random and torch.cuda) modules, respectively.
        # Remark: Use one seed for pytorch (same for torch.random and torch.cuda)
        self.seeds = {'random': random_seed, 'numpy': random_seed+1, 'torch': random_seed+2}

        # Set these random seeds for the different modules' random number generators
        random.seed(self.seeds['random'])
        np.random.seed(self.seeds['numpy'])
        # Remarks: 1) 'torch.manual_seed' calls 'torch.cuda.manual_seed_all' in the current pytorch version (torch==1.13.0).
        #          2) 'torch.cuda.manual_seed_all' itself calls 'torch.cuda.manual_seed' for each available GPU and thus seeds 
        #              all of them with the same seed 'self.seeds['torch]'.
        torch.manual_seed(self.seeds['torch'])

    def reset_states(self):
        """
        Reset the random state for the random-number-generators of the random, numpy, 
        and pytorch (torch.random and torch.cuda) modules back to their original values 
        before setting the random seed using 'set_seed'.
        """
        # Check that the attribute orig_states is not None
        if self.orig_states is None:
            err_msg = f"Can not reset the random state in case that not seed has been set so far (and thus self.orig_states is None)."
            raise AttributeError(err_msg)

        # Set the random states back to their initial values
        random.setstate(self.orig_states['random'])
        np.random.set_state(self.orig_states['numpy'])
        torch.cuda.set_rng_state_all(self.orig_states['torch_cuda_all'])
        torch.random.set_rng_state(self.orig_states['torch_random'])

        # Set the attributes orig_states and seeds to None
        self.orig_states = None
        self.seeds       = None

    def save_states(self):
        """
        Save the random state for the random-number-generators of the random, numpy, 
        and pytorch (torch.random and torch.cuda) modules.
        """
        # Check that the attribute orig_states is None
        if self.orig_states is not None:
            err_msg = f"Can not save a random seed in case that the the original state has not already been reset (and thus self.orig_states is not None)."
            raise AttributeError(err_msg)

        # Get the current random states of the rngs of the different modules and save
        # them as the 'original states' so that they can be reset later.
        self.orig_states = dict()
        self.orig_states['random']         = random.getstate()
        self.orig_states['numpy']          = np.random.get_state()
        self.orig_states['torch_cuda_all'] = torch.cuda.get_rng_state_all()
        self.orig_states['torch_random']   = torch.random.get_rng_state()

        # Set the attribute seeds to None
        self.seeds = None
