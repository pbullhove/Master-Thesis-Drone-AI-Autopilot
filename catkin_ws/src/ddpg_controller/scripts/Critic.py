from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lr_critic):
        """
        Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            lr_critic (float): Learning rate of critic optimizer
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model(lr_critic)

    def build_model(self, lr_critic):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values.
        """

        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # net = layers.BatchNormalization()(states)

        # Add hidden layer(s) for state pathway
        # net = layers.Dense(units=400, \
        #     activation='relu', \
        #         kernel_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'), \
        #             bias_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'), \
        #                 kernel_regularizer=regularizers.l2(1e-2))(states)
        net = layers.Dense(units=400, activation='relu')(states)
        
        # net = layers.Add()([net, actions])
        net = layers.Concatenate()([net, actions])
        
        # net = layers.Dense(units=300, \
        #     activation='relu', \
        #         kernel_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'), \
        #             bias_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'), \
        #                 kernel_regularizer=regularizers.l2(1e-2))(net)
        net = layers.Dense(units=300, activation='relu')(net)

        # Add final output layer to produce action values (Q values)
        # Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3), \
        #     bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3), \
        #         kernel_regularizer=regularizers.l2(1e-2))(net)
        Q_values = layers.Dense(units=1, name='q_values', kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3), \
            bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with
        # built-in loss function
        optimizer = optimizers.Adam(lr=lr_critic)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[self.model.input[0], self.model.input[1], K.learning_phase()],
            outputs=action_gradients)
