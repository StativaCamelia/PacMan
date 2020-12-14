import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Network:
    def __init__(self, input_size, action_size, isTrained):
        self.input_size = input_size
        self.action_size = action_size
        self.isTrained = isTrained
        self.model_file = './pacman.h5'
        self.model = self.create_q_model()
        self.future_model = self.create_q_model()
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

    def create_q_model(self):
        if self.isTrained:
            created_model = self.load_model()
        else:
            inputs = layers.Input(self.input_size)
            layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
            layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
            layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
            layer4 = layers.Flatten()(layer3)
            layer5 = layers.Dense(512, activation="relu")(layer4)
            layer6 = layers.BatchNormalization()(layer5)
            action = layers.Dense(self.action_size, activation="linear")(layer6)
            created_model = keras.Model(inputs=inputs, outputs=action)

        return created_model

    def save_model(self, file):
        self.model.save(file)

    def load_model(self):
        return keras.models.load_model(self.model_file, compile=False)


class Pacman:
    def __init__(self):
        self.max_memory_length = 100000
        self.gamma = 0.99
        self.learn_freq = 4
        self.update_future_freq = 2000
        self.epsilon_min = 0.1
        self.epsilon_update = 0.0000009
        self.epsilon_greedy_exploration = 1.0
        self.batch_size = 32
        self.env = gym.make('MsPacman-v0')
        self.env = wrap_deepmind(self.env, frame_stack=True, scale=True)
        self.episodes = 10200
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.network = Network(self.state_size, self.action_size, False)
        self.loss_function = keras.losses.mse
        self.optimizer = keras.optimizers.Adam(learning_rate=0.005, clipnorm=10)

    def train(self):
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        frame_count = 0
        best = 0
        for episode in range(self.episodes):
            total_reward = 0
            state = np.array(self.env.reset())
            # Skip first 30 frames when the game starts
            for _ in range(30):
                self.env.step(0)
            t = 0
            for t in range(100000):
                frame_count += 1
                # self.env.render()
                # Probability to randomly choose an action
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
                reward *= 10
                state_next = np.array(state_next)
                total_reward += reward
                # Buffer update
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next
                if len(done_history) > self.batch_size and frame_count > 10000 and frame_count % self.learn_freq == 0:
                    # Choose from buffer
                    indices = np.random.choice(range(len(done_history)), size=self.batch_size)
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
                    # Approximate sum of future rewards by predicting the next optimal action
                    future_rewards = self.network.future_model.predict(state_next_sample,
                                                                       callbacks=[self.network.early_stopping])
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                    # Labels
                    masks = tf.one_hot(action_sample, self.action_size)
                    # Feed-forward
                    with tf.GradientTape() as tape:
                        q_values = self.network.model(state_sample)
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        loss = self.loss_function(updated_q_values, q_action)
                    # Backpropagation
                    grads = tape.gradient(loss, self.network.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.network.model.trainable_variables))
                # Update future weights
                if frame_count > 10000 and frame_count % self.update_future_freq == 0:
                    self.network.future_model.set_weights(self.network.model.get_weights())
                # Remove from buffer
                if len(rewards_history) > self.max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]
                if done:
                    break
            # All rewards
            episode_reward_history.append(total_reward)
            if total_reward > best:
                self.network.save_model("./pacman_best.h5")
                best = total_reward
            if episode % 30 == 0:
                self.plot(episode, episode_reward_history)
            template = "running reward: {:.2f} at episode {},frames played {}, frame count total {}"
            print(template.format(total_reward, episode, t, frame_count))
        self.network.save_model("./pacman.h5")
        print("Training finished and models saved.")

    def plot(self, episode, episode_reward_history):
         episode_reward_history = [
             episode_reward_history[a] + episode_reward_history[a + 1] + episode_reward_history[a + 2] for a in
             range(0, len(episode_reward_history) - 2, 3)]
         plt.plot(range(episode // 3), episode_reward_history)
         plt.title('Training reward')
         plt.xlabel('Episodes')
         plt.ylabel('Reward')
         plt.legend()
         plt.draw()
         plt.savefig('plots/rewards_{}.png'.format(episode))
         # plt.show(block = False)

    def play(self):
        model = self.network.load_model()
        for i_episode in range(20):
            observation = self.env.reset()
            for t in range(10000):
                self.env.render()
                if self.epsilon_min > np.random.rand(1)[0]:
                    action = np.random.choice(self.action_size)
                else:
                    state_tensor = tf.convert_to_tensor(observation)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = model(state_tensor)
                    action = tf.argmax(action_probs[0]).numpy()
                observation, reward, done, info = self.env.step(action)
                if done:
                    break

if __name__ == "__main__":
    pac = Pacman()
    pac.train()
    # pac.play()