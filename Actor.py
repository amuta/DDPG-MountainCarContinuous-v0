from keras import layers, models, optimizers, initializers
from keras import backend as K
import numpy as np

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()
       
    def reset_weights(self):
        session = K.get_session()
        for layer in self.model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=16, activation='relu')(states)
        # out = layers.Dropout(0.1)(net)
        net = layers.Dense(units=4, activation='relu')(net)
        # out = layers.Dropout(0.1)(net)
        # net = layers.Dense(units=16, activation='relu')(net)


        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # set starting weights to a lower value
        # print(self.model.layers[3].get_weights()[0], flush=True)
        # self.normalize(0.1)
        # self.new_weights()
        # print('before', self.model.get_weights(), flush=True)
        # for layer in model.layers: layer.weights(layer.get_weights * 0)
        
        # print('after', self.model.get_weights(), flush=True)
        # raise ValueError('look at those weights.')

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)