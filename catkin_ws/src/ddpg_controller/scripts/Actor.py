from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, lr_actor):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            lr_actor (float): Learning rate of actor optimizer
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model(lr_actor)

    def build_model(self, lr_actor):
        """
        Build an actor (policy) network that maps states -> actions.
        """
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # net = layers.BatchNormalization()(states)
        
        # net = layers.Dense(units=400, \
        #     activation='relu', \
        #         kernel_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'), \
        #             bias_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'))(states)
        net = layers.Dense(units=400, activation='relu')(states)

        # net = layers.BatchNormalization()(net)

        # net = layers.Dense(units=300, \
        #     activation='relu', \
        #         kernel_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'), \
        #             bias_initializer=initializers.VarianceScaling(scale=1.0/3, mode='fan_in', distribution='uniform'))(net)

        net = layers.Dense(units=300, activation='relu')(net)

        # net = layers.BatchNormalization()(net)        

        # final output layer
        actions = layers.Dense(units=self.action_size, activation='tanh', name='raw_actions', \
            kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3), \
                bias_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(net)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # output1 = [layer.output for layer in self.model.layers]
        # print_func = K.function([self.model.input, K.learning_phase()],output1)
        # layer_outputs = print_func(inputs=[states, 1.])
        # print("hiyyyy",self.model.layers[1].output)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=lr_actor)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
