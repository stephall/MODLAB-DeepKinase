# model_handling.py

# Import public modules
import collections
import time
import torch
import re
import numpy as np

# Import custom modules
from . import model_storage
from . import utils
from . import random_handling

class ModelHandler(object):
    # Define the default optimizer configuration as dictionary     
    _default_optimizer_config = {
        'name': 'Adam',
        'params': {
            'lr': 1e-3,
            'weight_decay': 5e-4,
        },
    }

    # Define the default learning rate scheduler configuration as dictionary   
    # Remark: The default learning rate scheduler is an exponential learning rate scheduler 
    #         with gamma=1.0 that results in a constant learning rate.  
    _default_learning_rate_scheduler_config = {
        'name': 'ExponentialLR',
        'params': {
            'gamma': 1.0,
        },
    }

    def __init__(self, 
                 model, 
                 data_handler, 
                 logger=None, 
                 **config_dict):
        """
        Initialize model handler that is used to govern both the model, but also the data
        it is trained and evaluated on.
        
        Args:
            model (src.models.model_templates.BaseModel): Model object.
            data_handler (src.data_handling.DataHandler or None): Data handler object or None.
                If None, no data will be available and certain methods cannot be used.
            logger (logging.logger or None): Logger object or None.
                (Default: None)
            **config_dict (dict): Dictionary containing the configurations.

        """
        # Assign inputs to class attributes
        self.model        = model
        self.data_handler = data_handler
        self.logger       = logger
        self.config_dict  = config_dict
        self.display_info(self.config_dict)

        # Get a dictionary containing the data loaders (mapped to the device the model is on) 
        # and assign it to a class attribute
        if data_handler is None:
            self.dataloader_dict = {'train': None, 'valid': None, 'test': None}
        else:
            self.dataloader_dict = data_handler.get_dataloader_dict(device=self.model.device)

        #######################################################################################################
        # Parse certain configurations by assigning them to class attributes or use default values for the
        # class attributes if their corresponding key-value pairs are not in the configurations dictionary.
        # Also update the corresponding entries in the configurations dictionary.
        #######################################################################################################
        # Define the path in which the model parameters and tracked quantitites 
        # (collectively called 'checkpoints') should be saved in
        self.checkpoints_dir_path                = self.config_dict.get('checkpoints_dir_path', './checkpoints')
        self.config_dict['checkpoints_dir_path'] = self.checkpoints_dir_path

        # Define in which epoch intervals we should evaluate the metric
        self.eval_metric_epoch_step                = self.config_dict.get('eval_metric_epoch_step', 1)
        self.config_dict['eval_metric_epoch_step'] = self.eval_metric_epoch_step

        # Define in which epoch intervals we should save the model parameters
        self.save_model_epoch_step                = self.config_dict.get('save_model_epoch_step', 5)
        self.config_dict['save_model_epoch_step'] = self.save_model_epoch_step

        # Define the random seed used for metric evaluation
        self.training_random_seed                = self.config_dict.get('training_random_seed', 84)
        self.config_dict['training_random_seed'] = self.training_random_seed

        # Define the random seed used for metric evaluation
        self.eval_random_seed                = self.config_dict.get('eval_random_seed', 128)
        self.config_dict['eval_random_seed'] = self.eval_random_seed
        #########################################################################################

        # Initialize the epoch to 0 as we have not yet trained anything
        self.epoch = 0

        # Initialize the optimizer first to None and then define it right afterwards
        self.optimizer = None
        self.define_optimizer()

        # Initialize the learning_rate_scheduler first to None and then define it right afterwards
        self.learning_rate_scheduler = None
        self.define_learning_rate_scheduler()

        # Define a storage object passing the checkpoint directory path
        self._storage = model_storage.ModelStorage(self.checkpoints_dir_path)

        # Define a list containing tracked quantities and use it to initialize the dictionary that 
        # will contain lists for each tracked quantities in the form [(epoch, tracked_value), ...].
        # Remark: Tracked metrics should contain 'metric' in their name and tracked losses should 
        #         contain 'loss' in their name.
        #         Moreover, tracked quantities should not contain both 'metric' and 'loss' in their name!
        self.tracked            = ['running_loss_train', 'metric_train', 'metric_valid', 'learning_rate']
        self._tracked_list_dict = {tracked_quantity: list() for tracked_quantity in self.tracked}

        # Display the model and optimizer information
        self.display_info(f"\nModel:\n{self.model}\n")
        self.display_info(f"\nModel on device: {self.model.device}\n")
        self.display_info(f"\nOptimizer:\n{self.optimizer}\n")

        # Initialize different random handlers as class attribute
        self.training_random_handler = random_handling.RandomHandler() # For training (import when using dropout for example)
        self.eval_random_handler     = random_handling.RandomHandler() # For metric evaluation and prediction

    @property
    def model_dir_path(self):
        """ Return the path to the model's checkpoints directory. """
        return self._storage.checkpoints_dir_path

    def display_info(self, 
                     info_msg):
        """
        Display information either logging it or by printing it if no logger is defined (self.logger is None).

        Args:
            info_msg (str): Information to be displayed.
        
        """
        if self.logger is not None:
            self.logger.info(info_msg)
        else:
            print(info_msg)

    def display_model_information(self, 
                                  **kwargs):
        """
        Wrapper to display model information. 

        Args:
            **kwargs (dit): Forwarded to self.model.display_model_information.
        
        """
        # Try to display model information and if there is an attribute error, inform the user about the problem and continue.
        try:
            self.model.display_model_information(logger=self.logger, **kwargs)                    
        except AttributeError:
            self.display_info("Could not display model information as calling model method 'display_model_information' returned an attribute error.\nCheck if the method 'display_model_information' is implemented for the model (and sub-models).")
            pass

    def define_optimizer(self):
        """ Define the optimizer (i.e., its attributes). """
        # If the optimizer is passed as configuration, assign it to a variable 'optimizer_config' 
        # and else use the default optimizer configuration
        optimizer_config = self.config_dict.get('optimizer', self._default_optimizer_config)

        # Differ cases where the variable optimizer_config is a dictionary or not
        if isinstance(optimizer_config, dict):
            # Get a handle to the optimizer class by its name
            optimizer_handle = getattr(torch.optim, optimizer_config['name'])

            # Define the optimizer object and assign it to the corresponding class attribute
            self.optimizer = optimizer_handle(self.model.parameters(), **optimizer_config['params'])
        else:
            # Otherwise, throw an errow
            err_msg = f"The configuration 'optimizer' must be a dictionary containing the 'name' (->str) of the optimizer and its parameters 'params' (->dict) as keys, got type {type(optimizer_config)} instead."
            raise TypeError(err_msg)

    def define_learning_rate_scheduler(self, 
                                       last_epoch=-1):
        """
        Define the learning rate scheduler attribute. 

        Args:
            last_epoch (int): Last epoch before defining the learning rate scheduler.
                This allows to resume training at a certain epoch, while ensuring that
                the learning rate scheduler is set to the correct epoch.
        
        """
        # If the learning_rate_scheduler is passed as configuration, assign it to a variable 'learning_rate_scheduler_config' 
        # and else use the default learning_rate_scheduler configuration
        learning_rate_scheduler_config = self.config_dict.get('learning_rate_scheduler', self._default_learning_rate_scheduler_config)

        # Differ cases where the variable learning_rate_scheduler is a dictionary or not
        if isinstance(learning_rate_scheduler_config, dict):
            # Get a handle to the learning rate scheduler class by its name
            learning_rate_scheduler_handle = getattr(torch.optim.lr_scheduler, learning_rate_scheduler_config['name'])

            # Define the learning_rate_scheduler object and assign it to the corresponding class attribute
            self.learning_rate_scheduler = learning_rate_scheduler_handle(self.optimizer, last_epoch=self.epoch-1, **learning_rate_scheduler_config['params'])
        else:
            # Otherwise, throw an errow
            err_msg = f"The configuration 'learning_rate_scheduler' must be a dictionary containing the 'name' (->str) of the scheduler and its parameters 'params' (->dict) as keys, got type {type(learning_rate_scheduler_config)} instead."
            raise TypeError(err_msg)

    def eval_metric(self, 
                    set_name):
        """
        Evaluate the (average) metric for one of the sets ('train', 'valid', or 'test').

        Args:
            set_name (str): Specifies on which set ('train', 'valid', or 'test') the metric should
                be evaluated on.

        Return:
            (float): Evaluated (average) metric on one of the sets ('train', 'valid', or 'test').
        
        """
        # Check that 'set_name' is allowed
        if set_name not in self.dataloader_dict:
            err_msg = f"Got '{set_name}' for the input 'set_name', but it must be one of the following: {list(self.dataloader_dict.keys())}"
            raise KeyError(err_msg)

        # If the dataloader of the set is not defined (= is None), display a message to the user, and return np.nan as metric
        if self.dataloader_dict[set_name] is None:
            self.display_info(f"Cannot evaluate the metric on the '{set_name}' set because it does not contain any data \nand thus return np.nan as value for this metric. ")
            return np.nan

        # Store the initial setting if the model is training or not
        # Remark: 'self.model.training' is a boolean
        initial_model_training_flag = self.model.training

        # Set the model in evaluation mode
        self.model.eval()

        # Get the current state of the dataloader's generator before running over all batches
        # Remark: In case no generator was defined for the dataloader (deterministic dataloader),
        #         the generator attribute of the dataloader is None, in which case there is no initial
        #         state and we set it to None as well.
        if self.dataloader_dict[set_name].generator is None:
            # As the generator is None, set the dataloader init state also to None
            dataloader_init_state = None
        else:
            # Get the initial state of the dataloader
            dataloader_init_state = self.dataloader_dict[set_name].generator.get_state()

            # Set a random seed so that the order (and composition) of the batches will
            # always be the same when calling this method thereby rendering the dataloader
            # defacto deterministic for evaluation (for reproducibility).
            self.dataloader_dict[set_name].generator.manual_seed(0)

        # Set the random seed for metric evaluation (i.e. for the output predictions) using the evaluation random handler
        self.eval_random_handler.set_seed(self.eval_random_seed)

        # Loop over the batches of the set
        sum_metric  = 0.0
        num_samples = 0
        with torch.no_grad(): # Don't need to use gradients
            for batch_data in self.dataloader_dict[set_name]:
                # Evaluate the metric over the current batch
                metric_batch = self.model.metric(batch_data)

                # Add the batch metric to the total metric and add the number of samples in the 
                # batch, accessible with len(batch_data), to the number of samples.
                sum_metric  += metric_batch.item()
                num_samples += len(batch_data)

        # Reset the random states to initial values before evaluation
        self.eval_random_handler.reset_states()

        # Set the state of the dataloader's generator back to its initial value before
        # running over all batches
        # Remark: In case that a generator was defined for the dataloader, the dataloader's
        #         generator attribute is not None. Only set the state to its intial value
        #         if this is the case.
        if self.dataloader_dict[set_name].generator is not None:
            self.dataloader_dict[set_name].generator.set_state(dataloader_init_state)

        # Set the model back to training mode if it was initially in training mode 
        # (and thus the boolean flag 'initial_model_training_flag' is true)
        if initial_model_training_flag:
            # If the initial model training flag was to true, set the model into training mode
            self.model.train()

        # Calculate the average metric over all samples and return it
        return sum_metric/num_samples

    def _save(self):
        """ Save the model for the current epoch (self.epoch). """
        # Save the model function's parameters
        self._storage.save_model_parameters(self.model, self.epoch)

        # Save the dictionary containing the lists of the tracked quantities
        self._storage.save_tracked_list_dict(self._tracked_list_dict, self.epoch)
    
    def load(self, 
             epoch='last'):
        """
        Load the model for the input epoch.

        Args:
            epoch (int or str): Epoch to load the model from. This is either a positive integer or a str 'last'.
                (Default: 'last')

        Return:
            None
        
        """
        # Parse the epoch using the storage instance and assign the parsed epoch to the
        # corresponding class attribute
        self.epoch = self._storage.parse_epoch(epoch)

        # Load the model function's parameters
        self._storage.load_model_parameters(self.model, self.epoch)

        # Load the dictionary of lists of tracked quantities and assign it to the 
        # corresponding class attribute
        self._tracked_list_dict = self._storage.get_tracked_list_dict(self.epoch)

    def train(self, 
              num_epochs=200):
        """ 
        Train the model.

        Remark: Epoch indexing is one-based but batch indexing is zero-based.
        
        Args:
            num_epochs (int): Number of epochs the model should be trained for.
                (Default: 200)

        Return:
            None
        
        """
        # Cleanup the model's checkpoints directory
        self._storage.cleanup_checkpoints_dir()

        # Display training to user
        if self.epoch==0:
            self.display_info(f"Train model for {num_epochs} epochs...\n")
        else:
            #self.display_info(f"Start at epoch {self.epoch} and train model for {num_epochs} additional epochs...\n")
            raise NotImplementedError("Resuming of training is not implemented (i.e. 'train data loader random state', 'optimizer', and 'learning rate schedulers' loading have not been implemented yet for this purpose).")

        # Set the model into training mode
        self.model.train()

        # Set the training random seed
        self.training_random_handler.set_seed(self.training_random_seed)
        self.display_info(f"Set random seed {self.training_random_seed} for training (necessary for probabilistic model outputs i.e. when using dropout during training).\n")

        # Loop over epochs (epoch indexing is one-based)
        start_time = time.time()
        for epoch in range(self.epoch+1, self.epoch+1+num_epochs):
            # Set the epoch attribute to the current epoch
            self.epoch = epoch

            # Initialize sum_running_loss_train and num_samples to zero
            sum_running_loss_train = 0.0
            num_samples            = 0
            
            # Loop over batches (batch indexing is zero-based)
            for batch_data in self.dataloader_dict['train']:
                # Train on the current batch
                # Remark: '_train_on_batch' returns the loss value over the entire batch that should 
                #         be extensive (i.e. scale with the number of data points in the batch).
                loss_batch_value = self._train_on_batch(batch_data)

                # Add the batch loss value to sum_running_loss_train and the number of samples in the batch, 
                # accessible via len(batch_data), to num_samples.
                sum_running_loss_train += loss_batch_value
                num_samples            += len(batch_data)
            
            # Display the epoch summary
            self.display_info('-'*100)
            self.display_info(f"Epoch {self.epoch} summary:")

            # Calculate the average running train loss for this epoch
            running_loss_epoch_train = sum_running_loss_train/num_samples
            
            # Display the average running train loss for this epoch
            self.display_info(f"[{self.epoch}] Running loss (train): {running_loss_epoch_train}")
            
            # Append the epoch's running train loss to its corresponding list of tracked quantities
            self._tracked_list_dict['running_loss_train'].append( (epoch, running_loss_epoch_train) )

            # Evaluate and display the train and validation metric at certain epochs periodically
            if (self.epoch-1)%self.eval_metric_epoch_step==(self.eval_metric_epoch_step-1):
                ###################################################################################################
                # Display train metric
                ###################################################################################################
                # Evaluate the metric on the train set
                metric_epoch_train = self.eval_metric('train')

                # Append the epoch's train metric values to the corresponding tracked list
                self._tracked_list_dict['metric_train'].append( (epoch, metric_epoch_train) )

                # Display the train metric values for the current epoch
                # Remark: The spaces are used to make it match the output of the running loss on the train set
                self.display_info(f"[{self.epoch}] Metric (train):       {metric_epoch_train}")
                ###################################################################################################

                ###################################################################################################
                # Display validation metric
                ###################################################################################################
                # Evaluate the metric on the validation set
                metric_epoch_valid = self.eval_metric('valid')

                # Append the epoch's validation metric values to the corresponding tracked list
                self._tracked_list_dict['metric_valid'].append( (epoch, metric_epoch_valid) )

                # Display the validation metric values for the current epoch
                # Remark: The spaces are used to make it match the output of the running loss on the train set
                self.display_info(f"[{self.epoch}] Metric (valid):       {metric_epoch_valid}")
                ###################################################################################################

                ###################################################################################################
                # Display learning rate
                ###################################################################################################
                # Get the list of current learning rate
                # Remark: The list contains the learning rates for each parameter group
                learning_rate = self.learning_rate_scheduler.get_last_lr()

                # In case that there is only one parameter group, redefine the learning rate to this single value
                if len(learning_rate)==1:
                    learning_rate = learning_rate[0]

                # Append the learning_rate to the corresponding tracked list
                self._tracked_list_dict['learning_rate'].append( (epoch, learning_rate) )

                # Display the validation metric values for the current epoch
                # Remark: The spaces are used to make it match the output of the running loss on the train set
                self.display_info(f"[{self.epoch}] Learning rate:        {learning_rate}")
                ###################################################################################################

                # Display model information
                self.display_model_information()

            # Save the model for certain epochs
            if (self.epoch-1)%self.save_model_epoch_step==(self.save_model_epoch_step-1):
                self._save()

            # Update the learning rate
            self.learning_rate_scheduler.step()
            
            self.display_info('-'*100)
     
        self.display_info(f"\nTraining done (total training duration: {(time.time()-start_time)/60:.2f} min)")

        # After training reset the random states before training
        self.training_random_handler.reset_states()

        # Set the model into evaluation mode (because training is finished)
        self.model.eval()

        # Save the model
        self._save()

    def _train_on_batch(self, 
                        batch_data):
        """
        Train the model on the passed batch.

        Args:
            batch_data (pytorch or pytorch_geometric dataset): Data of the batch that should contain 
                the train-targets as attribute 'y' and be passable to the models 'forward' method.

        Returns:
            (float): The loss over the batch.

        """
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Calculate the loss on the batch
        # Remark: Ensure that the model is in training mode by calling its method '.train()'
        self.model.train()
        loss_batch = self.model.loss(batch_data)
        
        # Calculate the backward pass (and thereby set the gradients of the parameters)
        loss_batch.backward()
        
        # Update the parameters by performing an optimizer step
        self.optimizer.step()

        return loss_batch.item()

    def plot_learning_curve(self, 
                            epoch=None, 
                            **kwargs):
        """
        Plot the learning curve.
        
        Args:
            epoch (int, str or None): The epoch the learning curve should be plotted for.
                Can be either a positive integer, 'last', or None in which case the epoch
                attribute of the model will be used (self.epoch).
            **kwargs (dict): Forwarded to utils.plot_learning_curve.
        
        Return:
            None
        """
        # In case that epoch is None, use self.epoch
        if epoch is None:
            epoch = self.epoch

        # Call plot_learning_curve from the utils module
        utils.plot_learning_curve(self.checkpoints_dir_path, epoch=epoch, **kwargs)

    def predict_for_set(self, 
                        set_name, 
                        data_attributes=[], 
                        model_methods=[], 
                        random_seed=None):
        """
        Predict for one of the sets ('train', 'valid', or 'test').

        Args:
            set_name (str): Specifies on which set ('train', 'valid', or 'test') prediction
                should be performed on.
            data_attributes (list of str or): List of data attributes (as strings) or single string 
                that should be returned for each datapoint together with the prediction.
                (Default: [])
            model_methods (list of str or): List of model methods (as strings) or single string 
                that should be determined for the input together with the prediction.
                (Default: [])
            random_seed (None or int): The random seed to use for prediction in every batch.
                If None, 'self.eval_random_seed' will be used as prediction random seed.
                (Default: None)

        Return:
            (dict): Dictionary containing 'prediction' and the requested data attributes
                as dictionary-keys and their corresponding values as numpy.arrays.
        
        """
        # Check that 'set_name' is allowed
        if set_name not in self.dataloader_dict:
            err_msg = f"Got '{set_name}' for the input 'set_name', but it must be one of the following: {list(self.dataloader_dict.keys())}"
            raise KeyError(err_msg)

        # If the dataloader of the set is not defined (= is None), throw an errir
        if self.dataloader_dict[set_name] is None:
            err_msg = f"Cannot predict for the '{set_name}' set because it does not contain any data. "
            raise ValueError(err_msg)
            
        # In case that data_attributes is a string, transform it to a one element list
        if isinstance(data_attributes, str):
            data_attributes = [data_attributes]
        
        # Check that data_attributes is a list
        if not isinstance(data_attributes, list):
            err_msg = f"The input 'data_attributes' must be a list of strings, got type '{type(data_attributes)}' instead."
            raise TypeError(err_msg)

        # In case that model_methods is a string, transform it to a one element list
        if isinstance(model_methods, str):
            model_methods = [model_methods]
        
        # Check that model_methods is a list
        if not isinstance(model_methods, list):
            err_msg = f"The input 'model_methods' must be a list of strings, got type '{type(model_methods)}' instead."
            raise TypeError(err_msg)

        # Check that the entries of model_methods are strings that correspond to a method of the model
        for model_method in model_methods:
            # 1) Check that model_method is a string
            if not isinstance(model_method, str):
                err_msg = f"The input 'model_methods' must be a list of strings, got element '{model_method}' of type '{type(model_method)}' instead"
                raise TypeError(err_msg)

            # 2) Check that model_method corresponds to the name of a method of the model
            try:
                getattr(self.model, model_method)
            except AttributeError:
                err_msg = f"The element '{model_method}' of the input 'model_methods' does not correspond to the name of a model method."
                raise AttributeError(err_msg)

        # Parse and set random_seed
        if random_seed is None:
            # If random seed is not passed (and thus None) use self.eval_random_seed
            random_seed = self.eval_random_seed
        else:
            if isinstance(random_seed, (int, float)):
                # If the random seed is a number, make it an integer
                random_seed = int(random_seed)
            else:
                err_msg = f"The input 'random_seed' must be either None (=> use self.eval_random_seed) or a number, got type '{type(random_seed)}' instead."

        # Store the initial setting if the model is training or not 
        # Remark: 'self.model.training' is a boolean
        initial_model_training_flag = self.model.training

        # Set the model in evaluation mode
        self.model.eval()

        # Get the current state of the dataloader's generator before running over all batches
        # Remark: In case no generator was defined for the dataloader (deterministic dataloader),
        #         the generator attribute of the dataloader is None, in which case there is no initial
        #         state and we set it to None as well.
        if self.dataloader_dict[set_name].generator is None:
            # As the generator is None, set the dataloader init state also to None
            dataloader_init_state = None
        else:
            # Get the initial state of the dataloader
            dataloader_init_state = self.dataloader_dict[set_name].generator.get_state()

            # Set a random seed so that the order (and composition) of the batches will
            # always be the same when calling this method thereby rendering the dataloader
            # defacto deterministic for evaluation (for reproducibility).
            self.dataloader_dict[set_name].generator.manual_seed(0)

        # Set the random seed for predictions using the evaluation random handler
        self.eval_random_handler.set_seed(random_seed)

        # Loop over the batches of the set
        quantities_dict = collections.defaultdict(list)
        self.display_info(f"Predict for '{set_name}' set...\n")
        start_time = time.time()
        with torch.no_grad(): # Don't need to use gradients
            for batch_data in self.dataloader_dict[set_name]:
                #####################################################################################################################
                ### Extract the data attributes
                #####################################################################################################################
                # Get the data attribute values
                for data_attribute in data_attributes:
                    # Remark: If 'data_attribute' is not an attribute of batch_data, 'getattr' will throw an
                    #         AttributeError that is caught here and rethrown using a custom error message
                    try:
                        batch_data_attribute_values = getattr(batch_data, data_attribute)
                    except AttributeError:
                        # Determine all attributes of the batch data object
                        batch_data_attributes = re.findall(r'(\w*?)=', str(batch_data))
                        err_msg = f"The passed data attribute '{data_attribute}' is not an attribute of the torch batch data objects for the '{set_name}' set.\nPlease use one of the following attributes:\n{batch_data_attributes}"
                        raise AttributeError(err_msg)
                    
                    # Cast the attribute values of the batch to a numpy array
                    if torch.is_tensor(batch_data_attribute_values):
                        batch_data_attribute_values = batch_data_attribute_values.cpu().detach().numpy()
                    else:
                        batch_data_attribute_values = np.array(batch_data_attribute_values)
                        
                    # Append the attribute values of the batch to the corresponding list
                    quantities_dict[data_attribute].append(batch_data_attribute_values)
                    
                #####################################################################################################################
                ### Predict for the current batch
                #####################################################################################################################
                # Predict the model output for the current batch
                batch_prediction = self.model(batch_data)
                batch_prediction = batch_prediction.cpu().detach().numpy()
                if batch_prediction.ndim==0: # If scalar, make it 1D array with single entry
                    batch_prediction = batch_prediction.reshape(1)

                # Add the batch prediction values to corresponding list
                quantities_dict['prediction'].append(batch_prediction)

                # Evaluate the model methods for the current batch
                for model_method in model_methods:
                    # Get a handle to the model method
                    model_method_obj = getattr(self.model, model_method)

                    # Evaluate the method for the batch
                    batch_model_method_eval = model_method_obj(batch_data)
                    batch_model_method_eval = batch_model_method_eval.cpu().detach().numpy()
                    if batch_model_method_eval.ndim==0: # If scalar, make it 1D array with single entry
                        batch_model_method_eval = batch_model_method_eval.reshape(1)

                    # Add the prediction evaluation for the current model method to corresponding list
                    quantities_dict[model_method].append(batch_model_method_eval)

        self.display_info(f"Prediction done (Duration: {time.time()-start_time:.2f}s)\n")
                
        # Reset the random states to initial values before prediction
        self.eval_random_handler.reset_states()

        # Set the state of the dataloader's generator back to its initial value before
        # running over all batches
        # Remark: In case that a generator was defined for the dataloader, the dataloader's
        #         generator attribute is not None. Only set the state to its intial value
        #         if this is the case.
        if self.dataloader_dict[set_name].generator is not None:
            self.dataloader_dict[set_name].generator.set_state(dataloader_init_state)

        # Set the model back to training mode if it was initially in training mode 
        # (and thus the boolean flag 'initial_model_training_flag' is true)
        if initial_model_training_flag:
            # If the initial model training flag was to true, set the model into training mode
            self.model.train()
            
        # Stack the numpy arrays in the lists corresponding to the dictionary-keys of quantities_dict
        # and return the resulting dictionary
        return {key: np.concatenate(value) for key, value in quantities_dict.items()}
        
    def predict(self, data):
        """ Predict the model output for data. """
        # Set the 'training' flag of the model function to False
        raise NotImplementedError("Class method 'predict' has not been implemented yet.")
