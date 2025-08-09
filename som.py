import jax
import jax.numpy as jnp
import flax.nnx as nnx
import math

class SOM(nnx.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, height, width, input_size, num_iters, learning_rate, rngs=nnx.Rngs(0), verbose=False):
        """
        height: height of the lattice
        width: width of the lattice
        num_iters: an integer denoting the number of iterations undergone while training.
        input_size: the dimensionality of the training inputs.
        learning_rate: a number denoting the initial time(iteration no)-based learning rate. Default value is 0.1
        """
        #To check if the SOM has been trained
        self._trained = False

        #Assign required variables first
        self._height = height
        self._width = width
        self._radius = max(height/2.0, width/2.0)
        self._input_size = input_size
        self._learning_rate = learning_rate
        self._effective_learning_rate = learning_rate
        self._num_iters = num_iters
        self._time_constant = num_iters / math.log(self._radius)

        if verbose:
            print("SOM height: {}".format(self._height))
            print("SOM width: {}".format(self._width))
            print("SOM initial radius {}".format(self._radius))
            print("SOM input size {}".format(self._input_size))
            print("SOM learning rate {}".format(self._learning_rate))
            print("SOM number of iterations {}".format(self._num_iters))
            print("SOM time constant {}".format(self._time_constant))

        # VARIABLES AND CONSTANT FOR DATA STORAGE

        # Randomly initialized weightage vectors for all neurons,
        # stored together as a matrix Variable of size [height * width, input_size]
        self._weights = nnx.Variable(jax.random.normal(rngs.params(), (height*width, input_size)))
        #shape(self._weights) = (M, L)

        # Matrix of size [height * width, 2] for SOM grid locations of neurons
        self._locations = jnp.array([[i, j] for i in range(height) for j in range(width)], jnp.int32)
        # shape(self._locations) = (M, 2)

    @property
    def trained(self):
        return self._trained

    @trained.setter
    def trained(self, value):
        self._trained = value

    def _train(self, inputs):
        # To compute the Best Matching Unit given a vector
        # Basically calculates the Euclidean distance between every
        # neuron's weightage vector and the input, and returns the
        # index of the neuron which gives the least value

        inputs = jnp.expand_dims(inputs, axis=1)
        # shape(inputs) = (N, 1, L)

        weights = jnp.expand_dims(self._weights, axis=0)
        # shape(weights) = (1, M, L)

        bmu_index = jnp.argmin(jnp.sum(jnp.square(inputs-self._weights), -1), axis=1)
        # shape(bmu_index) = (N)

        # This will extract the location of the BMU based on the BMU's index
        bmu_loc = self._locations[bmu_index]
        # shape(bmu_loc) = (N, 2)

        # Generate a vector with learning rates for all neurons,
        # based on iteration number and location wrt BMU

        bmu_loc = jnp.expand_dims(bmu_loc, axis=1)
        # shape(bmu_loc) = (N, 1, 2)

        bmu_distance_square = jnp.sum(jnp.square(self._locations-bmu_loc), -1, keepdims=True)
        # shape(bmu_distance_square) = (N, M, 1)

        neigh_radius_square = jnp.square(self._neigh_radius)

        bmu_distance_mask = nnx.relu(jnp.sign(neigh_radius_square-bmu_distance_square))
        # shape(bmu_distance_mask) = (N, M, 1)

        learning_efficiency = self._effective_learning_rate * jnp.exp(-0.5 * bmu_distance_square / neigh_radius_square)
        # shape(learning_efficiency) = (N, M, 1)

        effective_learning_efficiency = learning_efficiency * bmu_distance_mask
        # shape(effective_learning_efficiency) = (N, M, 1)

        # Finally, the op that will use learning_rate_op to update
        # the weightage vectors of all neurons based on a particular input

        delta = inputs - self._weights
        # shape(delta) = (N, M, L)

        self._weights += jnp.mean(effective_learning_efficiency * delta, axis=0)
        
        outputs = self._weights[bmu_index]

        return outputs

    def _test(self, inputs, verbose=False):
        inputs = jnp.expand_dims(inputs, axis=1)
        # shape(inputs) = (N, 1, L)

        weights = jnp.expand_dims(self._weights, axis=0)
        # shape(weights) = (1, M, L)

        bmu_index = jnp.argmin(jnp.sum(jnp.square(inputs - self._weights), -1), axis=1)
        # shape(bmu_index) = (N)

        outputs = self._weights[bmu_index]

        if verbose:
            print("bmu_index: {}".format(bmu_index))
            print("weights: {}".format(outputs))
        
        return outputs

    def __call__(self, inputs, is_training=True, verbose=False):
        # shape(inputs) = (N, L)

        if is_training:
            # To compute the current learning rate and neighbourhood values based on current iteration number
            for i in range(self._num_iters):
                self._effective_learning_rate = self._learning_rate * jnp.exp(-i/self._num_iters)
                self._neigh_radius = self._radius * jnp.exp(-i/self._time_constant)
                outputs = self._train(inputs)            
        else:
            outputs = self._test(inputs, verbose)

        return outputs

    @property
    def weights(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            warnings.warn("SOM not trained yet")
        return self._weights

    def bmu(self, inputs):
        """
        Maps input vector to the relevant neuron in the SOM grid.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")

        index = jnp.argmin(jnp.sum(jnp.square(inputs-self._weights), -1))
        loc = [index // self._height, index % self._width]
        color = self._weights[index]

        return loc, color

    @property
    def image(self):
        import numpy as np
        if self._input_size != 1 and self._input_size != 3:
            raise AttributeError("input_size should be 1 or 3 for later retriving weighting image")
        min_elem = np.min(self._weights)
        max_elem = np.max(self._weights)
        output = (self._weights - min_elem) / (max_elem - min_elem) * 255

        return np.reshape(output, [self._height, self._width, self._input_size]).astype(np.uint8)


