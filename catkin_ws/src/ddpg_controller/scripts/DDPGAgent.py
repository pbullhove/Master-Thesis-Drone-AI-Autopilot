from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer
from OUNoise import OUNoise
import numpy as np
import keras

class DDPGAgent():
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, state_size, action_size, action_low, action_high):
        # self.task = task
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high

        # learning rates
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.lr_actor)
        self.actor_target = Actor(self.state_size, self.action_size, self.lr_actor)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.lr_critic)
        self.critic_target = Critic(self.state_size, self.action_size, self.lr_critic)

        # store model architecture of actor and critic locally
        # keras.utils.plot_model(self.actor_local.model, '/home/danie/catkin_ws/src/ddpg/src/actor.png', show_shapes=True)        
        # keras.utils.plot_model(self.critic_local.model, '/home/danie/catkin_ws/src/ddpg/src/critic.png', show_shapes=True)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Initialize OU noise
        self.noise = OUNoise(action_size=self.action_size)

        # Currently testing with Gaussian noise instead of OU. Parameters for Gaussian follow
        self.noise_mean = 0.0
        self.noise_stddev = 0.2

        # Initialize replay buffer
        self.buffer_size = 1e6
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Parameters for DDPG
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()

    def choose_action(self, state):
        """Returns actions for given state(s) as per current policy."""
        pure_action = self.actor_local.model.predict(state)[0]
        # add gaussian noise for exploration
        # noise = np.random.normal(self.noise_mean, self.noise_stddev, self.action_size)
        
        # add OU noise for exploration
        noise = self.noise.sample()

        # action = np.clip(pure_action + noise, self.action_low, self.action_high)
        # print("pure", pure_action)
        # print("noise", noise)
        # action = self.action_high * (pure_action + noise)
        # action = pure_action + noise
        action = np.clip(pure_action + noise, self.action_low, self.action_high)
        # print("action", action)
        return action.tolist()

    def store_transition(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def train_actor_and_critic(self):
        """
        Update policy and value parameters using given batch of experience
        tuples.
        """

        # if not enough transitions in memory, don't train!
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample() # sample a batch from memory

        # Convert experience tuples to separate arrays for each element
        # (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in transitions if e is not None])
        actions = np.array([
            e.action for e in transitions if e is not None]).astype(
            np.float32).reshape(-1, self.action_size)
        rewards = np.array([
            e.reward for e in transitions if e is not None]).astype(
            np.float32).reshape(-1, 1)
        dones = np.array([
            e.done for e in transitions if e is not None]).astype(
            np.uint8).reshape(-1, 1)
        next_states = np.vstack(
            [e.next_state for e in transitions if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states) #mu_marked in algo
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next]) #Q' in algo

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones) #y_i in algo
        critic_loss = self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        # print("action_gradients",action_gradients)
        # custom training function
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft-update target models
        # self.soft_update(self.critic_local.model, self.critic_target.model, self.tau)
        # self.soft_update(self.actor_local.model, self.actor_target.model, self.tau)
        self.soft_update_critic()
        self.soft_update_actor()

        return critic_loss

    def soft_update_actor(self):
        """Soft update model parameters."""
        local_weights = np.array(self.actor_local.model.get_weights())
        target_weights = np.array(self.actor_target.model.get_weights())

        assert len(local_weights) == len(
            target_weights), ('Local and target model parameters must have '
                              'the same size')

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        self.actor_target.model.set_weights(new_weights)

    def soft_update_critic(self):
        """Soft update model parameters."""
        local_weights = np.array(self.critic_local.model.get_weights())
        target_weights = np.array(self.critic_target.model.get_weights())

        assert len(local_weights) == len(
            target_weights), ('Local and target model parameters must have '
                              'the same size')

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        self.critic_target.model.set_weights(new_weights)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(
            target_weights), ('Local and target model parameters must have '
                              'the same size')

        new_weights = tau * local_weights + (1 - tau) * target_weights
        target_model.set_weights(new_weights)

        