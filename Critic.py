from keras import layers, models, optimizers
from keras import backend as K
import numpy as np

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def normalize(self, rate):
        for i in [2,3,4,5]:
            w_mean = np.mean([w for w in self.model.layers[i].get_weights()[0]])
            b_mean = np.mean([b for b in self.model.layers[i].get_weights()[1]])
            w_new = [w - (w - np.random.normal(w, rate)) for w in self.model.layers[i].get_weights()[0]]
            b_new = [b - (b - np.random.normal(b_mean, rate)) for b in self.model.layers[i].get_weights()[1]]
            K.set_value(self.model.layers[i].weights[0], w_new)
            K.set_value(self.model.layers[i].weights[1], b_new)

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=4, activation='relu')(states)
        net_states = layers.Dense(units=4, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=4, activation='relu')(actions)
        net_actions = layers.Dense(units=4, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('sigmoid')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        # print('len layers', len(self.model.layers), flush=True)
        # for i in range(len(self.model.layers)):
            # print('aeho', self.model.layers[i], flush=True)
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='cosine')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)