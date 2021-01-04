from copy import deepcopy

import gym
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
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
        # self.target_model = self.create_q_model()
        # self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

    def create_q_model(self):
        if self.isTrained:
            created_model = self.load_model("./pacman_best.h5")
        else:
            # print(self.input_size)
            inputs = layers.Input(self.input_size)
            # layer0 = layers.Conv2D(64, 8, strides=1, activation="relu")(inputs)
            layer1 = layers.Conv2D(32, 8, strides=4, padding="same", activation="relu")(inputs)
            # layer1prim = layers.Conv2D(64, 8, strides=2, activation="relu")(layer1)
            layer2 = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(layer1)
            layer3 = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")(layer2)
            layer4 = layers.Flatten()(layer3)
            layer5 = layers.Dense(512, activation="relu")(layer4)
            # layer6 = layers.BatchNormalization()(layer5)
            action = layers.Dense(self.action_size, activation="linear")(layer5)
            created_model = keras.Model(inputs=inputs, outputs=action)
            created_model.compile(optimizer="Adam", loss='mse')

        return created_model

    def save_model(self, file):
        self.model.save(file)

    def load_model(self, file):
        return keras.models.load_model(file, compile=False)


class Pacman:
    def __init__(self):
        self.max_memory_length = 100000
        self.gamma = 0.99
        self.learn_freq = 4
        self.update_future_freq = 10000
        self.epsilon_min = 0.1
        # self.epsilon_update = 0.0000009
        self.epsilon_update = 1000000.0
        self.epsilon_greedy_exploration = 1
        self.batch_size = 128
        self.env = gym.make('MsPacman-v0')
        # self.env = wrap_deepmind(self.env, frame_stack=True, scale=False)
        self.episodes = 10200
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.network = Network((88, 80, 4), self.action_size, False)
        self.loss_function = keras.losses.MSE
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025)

        self.network.save_model("./pacman_init.h5")
        # self.optimizer = keras.optimizers.SGD(learning_rate=0.00001)
        # self.optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, clipnorm=10)

    def preprocess(self, obs):
        color = np.array([210, 164, 74]).mean()
        # Crop and resize the image
        img = obs[1:176:2, ::2]
        # Convert the image to greyscale
        img = img.mean(axis=2)
        # Improve image contrast
        img[img == color] = 0
        # Next we normalize the image from -1 to +1
        img = (img - 128) / 128 - 1
        return img.reshape(88, 80)

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
            state = self.preprocess(state)
            statelist = []

            # state = state * 2.0 - 1.0
            # state = self.preprocess(state)
            # Skip first 30 frames when the game starts
            for _ in range(50):
                self.env.step(0)

            while len(statelist) < 4:
                statelist.append(state)
                state_next, reward, done, _ = self.env.step(0)
                state_next = np.array(state_next)
                state_next = self.preprocess(state_next)
                state = state_next

            state = np.stack(statelist, axis=2)
            print(state.shape)

            t = 0
            done = False
            while not done:
                t += 1
                if t > 10000:
                    break
                frame_count += 1
                # self.env.render()
                # print(np.shape(state))
                # print(np.min(state))
                # print(np.max(state))
                # if frame_count == 50:
                #     plt.imshow(state[:, :, 0], cmap="gray")
                #     plt.show()
                # Probability to randomly choose an action

                if self.epsilon_greedy_exploration > np.random.rand(1)[0]:
                    action = np.random.choice(self.action_size)
                    # print(self.epsilon_greedy_exploration)
                else:
                    action_probs = self.network.model.predict(np.array([state]))
                    action = np.argmax(action_probs[0])
                    # print("action:", action)
                # self.epsilon_greedy_exploration *= 0.99
                # self.epsilon_greedy_exploration = max(self.epsilon_greedy_exploration, self.epsilon_min)
                state_next, reward, done, _ = self.env.step(action)
                # reward /= 10000.0
                # if done:
                #     reward = -1000
                state_next = np.array(state_next)
                state_next = self.preprocess(state_next)
                statelist.append(state_next)
                del statelist[:1]
                state_next = np.stack(statelist, axis=2)

                # state_next = self.preprocess(state_next)
                # state_next = state_next * 2.0 - 1.0
                total_reward += reward
                # Buffer update
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next
                if len(done_history) > self.batch_size and frame_count % self.learn_freq == 0 and frame_count > 10000:
                    # print("Learning")
                    # Choose from buffer
                    indices = np.random.choice(range(len(done_history)), size=self.batch_size)
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = np.array([rewards_history[i] for i in indices])
                    action_sample = np.array([action_history[i] for i in indices], dtype=np.int8)
                    done_sample = np.float32(np.array([float(done_history[i]) for i in indices]))

                    # Approximate sum of future rewards by predicting the next optimal action

                    action_onehot = tf.one_hot(action_sample, self.action_size, dtype=np.int8).numpy()
                    action_space = np.array([i for i in range(self.action_size, )], dtype=np.int8)
                    action_indices = np.dot(action_onehot, action_space)

                    q_values = self.network.model.predict(state_sample)
                    future_rewards = self.network.model.predict(state_next_sample)

                    q_target = q_values.copy()

                    # print(action_indices)
                    # print(np.shape(action_indices))
                    # print(rewards_sample.shape)
                    # print(future_rewards.shape)
                    # print(q_target.shape)

                    q_target[:, action_indices] = rewards_sample + self.gamma * np.max(future_rewards, axis=1) * (
                            1 - done_sample)
                    print(np.shape(q_target))

                    _ = self.network.model.fit(state_sample, q_target, verbose=0)

                    self.epsilon_greedy_exploration *= 0.99
                    self.epsilon_greedy_exploration = max(self.epsilon_greedy_exploration, self.epsilon_min)

                #     updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)
                #     updated_q_values = updated_q_values * (1 - done_sample) - 1000.0 * done
                #     # Labels
                #     masks = tf.one_hot(action_sample, self.action_size)
                #     updated_q_values = tf.expand_dims(updated_q_values, 1)
                #     print(masks)
                #     # Feed-forward
                #     with tf.GradientTape() as tape:
                #         q_values = self.network.model(state_sample)
                #
                #         # ys = tf.multiply(updated_q_values, masks)
                #         # ys = tf.add(ys, q_values)
                #
                #         q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                #
                #         # print("updated_q_values: ", updated_q_values)
                #         # print("q_action: ", q_action)
                #         # loss = tf.keras.losses.mean_squared_error(ys, q_values)
                #
                #         loss = tf.reduce_mean(tf.square(updated_q_values - q_action))
                #         # print("loss", loss)
                #         # Backpropagation
                #         grads = tape.gradient(loss, self.network.model.trainable_variables)
                #     self.optimizer.apply_gradients(zip(grads, self.network.model.trainable_variables))
                #
                # # Update future weights
                # if frame_count % self.update_future_freq == 0:
                #     self.network.target_model.set_weights(self.network.model.get_weights())
                # Remove from buffer
                if len(rewards_history) > self.max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]
            # All rewards
            episode_reward_history.append(total_reward)
            if total_reward >= best:
                self.network.save_model("./pacman_best.h5")
                best = total_reward
            if episode % 30 == 0:
                self.plot(episode, episode_reward_history)

            template = "running reward: {:.2f} at episode {},frames played {}, frame count total {}"
            print(template.format(total_reward, episode, t, frame_count))

            if episode % 15 == 0 and episode != 0:
                self.play_game(self.network.model)

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

    def play_game(self, model=None):
        model = self.network.load_model("pacman_best.h5")
        for i_episode in range(20):
            observation = np.array(self.env.reset())

            observation = self.preprocess(observation)
            statelist = []

            # state = state * 2.0 - 1.0
            # state = self.preprocess(state)
            # Skip first 30 frames when the game starts
            for _ in range(50):
                self.env.step(0)

            while len(statelist) < 4:
                statelist.append(observation)
                state_next, reward, done, _ = self.env.step(0)
                state_next = np.array(state_next)
                state_next = self.preprocess(state_next)
                observation = state_next

            observation = np.stack(statelist, axis=2)
            # print(observation.shape)

            done = False
            while not done:
                # observation = observation * 2.0 - 1.0
                # observation = self.preprocess(observation)
                # plt.imshow(observation[:, :, 0], cmap="gray")
                # plt.show()
                self.env.render()
                if 0.05 > np.random.rand(1)[0]:
                    action = np.random.choice(self.action_size)
                else:
                    # state_tensor = tf.convert_to_tensor(observation)
                    # state_tensor = tf.expand_dims(state_tensor, 0)
                    # print(observation.shape)
                    action_probs = model.predict(np.array([observation]))
                    action = np.argmax(action_probs[0])
                    # print(action)
                    # with open("actions" + str(i_episode) + ".txt", 'a') as f:
                    #     f.write(str(action) + ",")
                nextobservation, reward, done, info = self.env.step(action)
                observation = deepcopy(nextobservation)
                observation = np.array(observation)
                observation = self.preprocess(observation)
                statelist.append(observation)
                del statelist[:1]
                observation = np.stack(statelist, axis=2)



if __name__ == "__main__":
    pac = Pacman()
    pac.train()
    # pac.play_game()
