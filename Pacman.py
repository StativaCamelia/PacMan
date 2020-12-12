import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Network:
	def __init__(self, input_size, action_size):
		self.input_size = input_size
		self.action_size = action_size
		self.model = self.create_q_model()
		self.future_model = self.create_q_model()

	def create_q_model(self):
		inputs = layers.Input(self.input_size)
		layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
		layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
		layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
		layer4 = layers.Flatten()(layer3)
		layer5 = layers.Dense(512, activation="relu")(layer4)
		action = layers.Dense(self.action_size, activation="linear")(layer5)
		return keras.Model(inputs=inputs, outputs=action)

class Pacman:
	def __init__(self):
		self.max_memory_length = 100000
		self.gamma = 0.99
		self.learn_freq = 4
		self.epsilon_min = 0.1
		self.epsilon_update = 0.0000009
		self.epsilon_greedy_exploration = 1.0
		self.batch_size = 32
		self.env = gym.make('MsPacman-v0')
		self.env = wrap_deepmind(self.env, frame_stack=True, scale=True)
		self.episodes = 30
		self.state_size = self.env.observation_space.shape
		self.action_size = self.env.action_space.n
		self.network = Network(self.state_size, self.action_size)
		self.loss_function = keras.losses.Huber()
		self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

	def play(self):
		action_history = []
		state_history = []
		state_next_history = []
		rewards_history = []
		done_history = []
		episode_reward_history = []
		for i_episode in range(self.episodes):
			total_reward = 0
			frame_count = 0
			state = np.array(self.env.reset())
			for t in range(100000):
				frame_count += 1
				# self.env.render()
				if self.epsilon_greedy_exploration > np.random.rand(1)[0]:
					action = np.random.choice(self.action_size)
				else:
					state_tensor = tf.convert_to_tensor(state)
					state_tensor = tf.expand_dims(state_tensor, 0)
					action_probs = self.network.model(state_tensor)
					action = tf.argmax(action_probs[0]).numpy()
				self.epsilon_greedy_exploration -= self.epsilon_greedy_exploration * self.epsilon_update
				self.epsilon_greedy_exploration = max(self.epsilon_greedy_exploration, self.epsilon_min)
				state_next, reward, done, _ = self.env.step(action)
				state_next = np.array(state_next)
				total_reward += reward
				action_history.append(action)
				state_history.append(state)
				state_next_history.append(state_next)
				done_history.append(done)
				rewards_history.append(reward)
				state = state_next
				if len(done_history) > self.batch_size:
					indices = np.random.choice(range(len(done_history)), size=self.batch_size)
					state_sample = np.array([state_history[i] for i in indices])
					state_next_sample = np.array([state_next_history[i] for i in indices])
					rewards_sample = [rewards_history[i] for i in indices]
					action_sample = [action_history[i] for i in indices]
					done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
					future_rewards = self.network.future_model.predict(state_next_sample)
					updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
						future_rewards, axis=1
					)
					updated_q_values = updated_q_values * (1 - done_sample) - done_sample
					masks = tf.one_hot(action_sample, self.action_size)
					with tf.GradientTape() as tape:
						q_values = self.network.model(state_sample)
						q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
						loss = self.loss_function(updated_q_values, q_action)
					grads = tape.gradient(loss, self.network.model.trainable_variables)
					self.optimizer.apply_gradients(zip(grads, self.network.model.trainable_variables))
				if frame_count > 1000 and frame_count % self.learn_freq == 0:
					self.network.future_model.set_weights(self.network.model.get_weights())
				if len(rewards_history) > self.max_memory_length:
					del rewards_history[:1]
					del state_history[:1]
					del state_next_history[:1]
					del action_history[:1]
					del done_history[:1]
				if done:
					break
			template = "running reward: {:.2f} at episode {}, frame count {}"
			print(template.format(total_reward, i_episode, frame_count))
if __name__ == "__main__":
	pac = Pacman()
	pac.play()