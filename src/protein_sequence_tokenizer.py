# protein_sequence_tokenizer.py

# Import modules
import math
import itertools

# Define a class to tokenize protein (amino acid) sequences
class ProteinSequenceTokenizer(object):
    # Define a dictionary that maps amino acid symbol to amino acid name
    _symbol_to_name_map = {
        'A': 'Alanine',
        'G': 'Glycine',
        'I': 'Isoleucine',
        'L': 'Leucine',
        'P': 'Proline',
        'V': 'Valine',
        'F': 'Phenylalanine',
        'W': 'Tryptophan',
        'Y': 'Tyrosine',
        'D': 'Aspartic Acid',
        'E': 'Glutamic Acid',
        'R': 'Arginine',
        'H': 'Histidine',
        'K': 'Lysine',
        'S': 'Serine',
        'T': 'Threonine',
        'C': 'Cysteine',
        'M': 'Methionine',
        'N': 'Asparagine',
        'Q': 'Glutamine',
    }

    # Define the string symbol for zero padding
    _zero_padding_symbol = '<ZERO>'

    def __init__(self, 
                 k=1, 
                 stride=None, 
                 sequence_length=None, 
                 sequence_indel_strategy='back'):
        """
        Define a protein (amino acid) sequence tokenizer

        Args:
            k (int): Number defining which k-mer should be created, which must be bigger
                or equal to 1.
                (Default: 1)
            stride (int): Stride for k-merization, which must be an integer in [1, k].
                In case None is passed the stride is set to k.
                (Default: None)
            sequence_length (int): Length to which the sequences should be either restricted 
                or zero-padded to. If None, the sequences will neither be restricted nor padded.
                (Default: None)
            sequence_indel_strategy (str): Sequence insertion-or-deletion (indel) strategy used
                to either restrict or zero-pad sequences to the demanded length 'self.sequence_length'
                in case that it passed (thus 'self.sequence_length' is not None).
                The options are 'back', 'front', and 'symmetric'.
                (Default: 'back')
        """
        #############################################################################################################
        # Parse the inputs
        #############################################################################################################
        # 1) Parse k
        k = int(k)
        if k<1:
            err_msg = f"The input 'k' must be bigger or equal to 1, got value '{k}' instead."
            raise ValueError(err_msg)
        
        self.k = k

        # 2) Parse the stride
        if stride is None:
            stride = self.k

        stride = int(stride)
        if not (1<=stride<=self.k):
            err_msg = f"The input 'stride' must be an integer in [1, k={self.k}], got value '{stride}' instead."
            raise ValueError(err_msg)
        
        self.stride = stride

        # 3) Parse the sequence length
        if sequence_length is not None and sequence_length<1:
            err_msg = f"The input 'sequence_length' must be bigger or equal to 1, got value '{sequence_length}' instead."
            raise ValueError(err_msg)

        self.sequence_length = sequence_length

        # 4) Parse the sequence insertion-or-deletion (indel) strategy
        if sequence_indel_strategy not in ['back', 'front', 'symmetric']:
            err_msg = f"The input 'sequence_indel_strategy' must be 'back', 'front', or 'symmetry', got value '{sequence_indel_strategy}' instead."
            raise ValueError(err_msg)

        self.sequence_indel_strategy = sequence_indel_strategy
        #############################################################################################################

        # Generate the k-mer alphabet
        self.k_mer_alphabet = self._generate_k_mer_alphabet()

        # Generate the k-mer letter to numeral
        self.k_mer_letter_to_numeral_map = self._generate_k_mer_letter_to_numeral_map()

    def _generate_k_mer_alphabet(self):
        """
        Generate the k-mer alphabet based on self.k and the alphabet of amino acid symbols.
        
        Args:
            None

        Return:
            (list): Sorted k-mer alphabet.

        """
        # The basic (1-mer) alphabet corresponds to the set of amino acid sequences.
        # which are the keys of 'self._symbol_to_name_map'
        basic_alphabet = list(self._symbol_to_name_map.keys())

        # Sort the basic alphabet
        basic_alphabet.sort()

        # Build up the k-mer alphabet by inductively generating the n-mer alphabet from 
        # the basic alphabet and the (n-1)-mer alphabet using their cartesian product.
        # The n=1-mer alphabet correspond to the basic alphabet
        n_mer_alphabet = basic_alphabet

        # Do (n-1)->n steps to succesively build the n-mer alphabets until n=k
        for n in range(2, self.k+1):
            # Make the cartesian product of the basic alphabet and the (n-1)-mer alphabet
            # which will result in a list of 2-tuples.
            # E.g. The cartesian product of the basic alphabet ['A', 'B'] and the 2-mer alphabet 
            # ['AA', 'AB', 'BA', 'BB'] will result in the cartesian product
            # [('A', 'AA'), ('B', 'AA'), ('A', 'AB'), ('B', 'AB'), ...]
            cartesian_product = list( itertools.product(basic_alphabet, n_mer_alphabet) )
            
            # Join the strings of the 2-tuples to generate the n-mer alphabet
            # E.g. ('A', 'AA') to 'AAA'
            n_mer_alphabet = [''.join(two_tuple) for two_tuple in cartesian_product]
            

        # After these iterations, the sought for k-mer alphabet corresponds to the n-mer alphabet
        k_mer_alphabet = n_mer_alphabet

        # Sort the elements of the k-mer alphabet
        k_mer_alphabet.sort()

        # Check that the number of elements of the k-mer alphabet correspond to the expected number
        # given by N^k where N is the number of elements in the basic alphabet
        if len(k_mer_alphabet)!=int(len(basic_alphabet)**self.k):
            err_msg = f"The number of elements of the constructed k-mer alphabet ({len(k_mer_alphabet)}) is not equivalent " \
                      f"to the number of expected elements ({len(basic_alphabet)}^{self.k}={int(len(basic_alphabet)**self.k)})"
            raise ValueError(err_msg)

        return k_mer_alphabet

    def _generate_k_mer_letter_to_numeral_map(self):
        """ Return a map (dictionary) from the k-mer letter (element of k-mer alphabet) to their integer representation. """
        # Initialize a dictionary containing the attribute 'self._zero_padding_symbol' as key with value 0
        k_mer_letter_to_numeral_map = {self._zero_padding_symbol: 0}

        # Generate dictionary mapping the letter (element of k-mer alphabet) to its (one-based) index in the alphabet
        # and use it to update the previously defined dictionary.
        k_mer_letter_to_numeral_map.update( {k_mer_letter: index+1 for index, k_mer_letter in enumerate(self.k_mer_alphabet)} )

        return k_mer_letter_to_numeral_map

    def __call__(self, 
                 sequence):
        """
        Tokenize the input sequence that involves the steps:
            1) k-merization
            2) Restriction or zero-padding to demanded sequence length [optional]
            3) Mapping to numeric representation

        Args:
            sequence (str): Protein (amino acid) sequence.

        Return:
            (list): Numerically tokenized sequence.
        """
        # 1) k-merize the sequence
        k_mer_sequence = self._k_merize_sequence(sequence)

        # 2) Restrict or pad the k-merized sequence to the demanded length in case the sequences
        # should have the length 'self.sequence_length' (and thus 'self.sequence_length' is not None). 
        if self.sequence_length is not None:
            # Differ cases depending on the k-mer sequence length
            if self.sequence_length<len(k_mer_sequence):
                # In case that the k-mer sequence length is longer than the demanded length,
                # restrict it to this length.
                k_mer_sequence = self._restrict_sequence(k_mer_sequence)
            else:
                # In case the the k-mer sequence length is smaller than the demanded length, 
                # zero pad the k-mer sequence to this length.
                k_mer_sequence = self._zero_pad_sequence(k_mer_sequence)

        # 3) Map all k-mer letters of the k-merized sequence to their corresponding numeral
        numeric_sequence = [self.k_mer_letter_to_numeral_map[token] for token in k_mer_sequence]

        return numeric_sequence

    def _k_merize_sequence(self, 
                           sequence):
        """
        k-merize the input sequence. 
        
        Args:
            sequence (str): Protein (amino acid) sequence.

        Return:
            (list): List of tokens (strings) of the k-merized protein sequence.
        
        """
        # Generate the range of starting indices that correspond to the index of
        # the elements in the sequence with which each k-mer token will start with.
        starting_index_range  = range(0, len(sequence)-(self.k-1), self.stride)
        
        # Generate the k-mer sequence by iterating over the starting indices and generate each 
        # k-mer as the k sequential elements in the sequence that start with each starting index.
        return [sequence[starting_index:starting_index+self.k] for starting_index in starting_index_range]
    
    def _restrict_sequence(self, 
                           sequence):
        """
        Restrict the sequence to a length corresponding to 'self.sequence_length' and return it. 
        
        Args:
            sequence (list): List of tokens (strings) of a protein sequence.

        Return:
            (list): Restricted sequence (as list of string tokens).
        
        """
        # Determine the total number of elements to be deleted from the sequence
        num_deletions_total = len(sequence) - self.sequence_length

        # Determine the number of elements to be deleted at the front and the back
        num_deletions_front, num_deletions_back = self._get_number_of_indels(num_deletions_total)

        # Only keep sequence elements in the not-deleted range and return the resulting restricted sequence
        # Remark: The case where 'num_deletions_back' is zero must be treated explicitly/extra, because
        #         sequence[...:(-0)]=sequence[...:0] will result in an empty sequence.
        if 0==num_deletions_back:
            return sequence[num_deletions_front:]
        else:
            return sequence[num_deletions_front:-num_deletions_back]
        
    def _zero_pad_sequence(self, 
                           sequence):
        """ 
        Zero pad the input sequence to a length corresponding to 'self.sequence_length' and return it. 

        Args:
            sequence (list): List of tokens (strings) of a protein sequence.

        Return:
            (str): Zero-padded sequences (as list of string tokens).
        
        """
        # Determine the total number of zero pads to be inserted
        num_insertions_total = self.sequence_length-len(sequence)

        # Determine the number of zero pads to be inserted at the front and the back
        num_insertions_front, num_insertions_back = self._get_number_of_indels(num_insertions_total)

        # Generate the zero padded sequences (zps) to be added in the front and the back of the sequence
        zps_front = [self._zero_padding_symbol]*num_insertions_front
        zps_back  = [self._zero_padding_symbol]*num_insertions_back

        # Zero pad the sequence in the front as well as in the back and return it
        return zps_front + sequence + zps_back

    def _get_number_of_indels(self, 
                              num_indels_total):
        """
        Return the number of insertion-or-deletion (indels) elements for the front and back 
        of a sequence depending on 'self.sequence_indel_strategy'.

        Arg:
            num_indels_total (int): Total number of insertion-or-deletion (indel) elements.

        Return:
            (int, int): Number of (front, back) insertion-or-deletion (indel) elements.
        """
        # Different strategies to obtain the number of front and back indels
        if self.sequence_indel_strategy=='back':
            # None in the front and 'all' in the back
            num_indels_front = 0
            num_indels_back  = num_indels_total
        elif self.sequence_indel_strategy=='front':
            # None in the back and 'all' in the front
            num_indels_front = num_indels_total
            num_indels_back  = 0
        else: # Case: 'symmetric'
            # Remark: Because the value of 'self.sequence_indel_strategy' is checked when parsing 
            #         it as input to the constructor, it must be 'symmetric'.
            # Half in the front and half in the back
            num_indels_front = int( math.floor(num_indels_total/2) )
            num_indels_back  = num_indels_total - num_indels_front

        return num_indels_front, num_indels_back