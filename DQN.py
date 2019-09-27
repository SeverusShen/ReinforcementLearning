import numpy as np
import tensorflow as tf
from reinforcement_learning.RL import RL
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import sys


class DQN(RL):
    def __init__(self, maze, hidden_dim=16, cache_size=2000, batch_dim=32, n_actions=4, n_features=2,
                 eps=0, gamma=0.9, lr=0.001, pixel=100, draw=False, save_fn=None):
        super().__init__(maze, eps, gamma, lr, pixel, draw, save_fn)
        self.hidden_dim = hidden_dim
        self.cache_size = cache_size
        self.batch_dim = batch_dim
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon_increment = 0.001  # greedy algorithms
        self.epsilon_max = 0.9
        self.ddqn = True

        # memory cache
        self.cache_state_idxs = np.zeros((cache_size, self.n_features))
        self.cache_next_state_idxs = np.zeros((cache_size, self.n_features))
        self.cache_actions = np.zeros((cache_size,), dtype=int)
        self.cache_reward = np.zeros((cache_size,))

        # ------------------ build eval_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # inputs
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names = ('eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES)
            cw_initializer = tf.random_normal_initializer(0., 0.3)
            cb_initializer = tf.constant_initializer(0.1)
            self.q_eval = self.build_network(self.s, c_names, hidden_dim, cw_initializer, cb_initializer)
        with tf.variable_scope('loss'):
            # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_eval)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # inputs
        with tf.variable_scope('target_net'):
            t_names = ('target_net_params', tf.GraphKeys.GLOBAL_VARIABLES)
            tw_initializer = tf.random_normal_initializer(0., 0.3)
            tb_initializer = tf.constant_initializer(0.1)
            self.q_next = self.build_network(self.s_, t_names, hidden_dim, tw_initializer, tb_initializer)

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def build_network(self, inputs, name, hidden_dim, w_initializer, b_initializer):
        # first layer. collections is used later when assign to target net
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', (self.n_features, hidden_dim), initializer=w_initializer, collections=name)
            b1 = tf.get_variable('b1', (hidden_dim,), initializer=b_initializer, collections=name)
            l1 = tf.nn.relu(tf.nn.xw_plus_b(inputs, w1, b1))

        # second layer. collections is used later when assign to target net
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', (hidden_dim, self.n_actions), initializer=w_initializer, collections=name)
            b2 = tf.get_variable('b2', (self.n_actions,), initializer=b_initializer, collections=name)
            return tf.nn.xw_plus_b(l1, w2, b2)

    def train(self, itera, start_idx=(0, 0)):
        self.action_idx = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        step = 0
        for i in tqdm(range(itera)):
            state_idx = start_idx
            while self.maze[state_idx] not in [1, 2]:
                s = np.array(state_idx).reshape((-1, self.n_features)) / self.maze.shape[0]
                q_eval = self.sess.run(self.q_eval, feed_dict={self.s: s})

                # get state
                state = {k: q_eval[0][k] for k in range(self.n_actions)}
                if state_idx[0] == 0:  # no up
                    state.pop(0)
                if state_idx[0] == self.maze.shape[0] - 1:  # no down
                    state.pop(1)
                if state_idx[1] == 0:  # no left
                    state.pop(2)
                if state_idx[1] == self.maze.shape[1] - 1:  # no right
                    state.pop(3)

                action = self.choose(state)
                idx = self.action_idx[action]
                next_state_idx = tuple((a + b for a, b in zip(state_idx, idx)))
                s_ = np.array(next_state_idx).reshape((-1, self.n_features)) / self.maze.shape[0]

                reward = self.get_reward(self.maze[next_state_idx])

                idx = step % self.cache_size
                self.cache_state_idxs[idx] = s.reshape((self.n_features, ))
                self.cache_next_state_idxs[idx] = s_.reshape((self.n_features, ))
                self.cache_actions[idx] = action
                self.cache_reward[idx] = reward

                state_idx = next_state_idx

                if step > self.batch_dim * 10 and step % 10 == 0:
                    self.learn(step)

                step += 1

                if i >= 20 and self.draw and i % (itera // 10) == 0:
                    maze = self.maze.copy()
                    maze[state_idx] = 4
                    self.show(maze, self.pixel, i)

                # greedy algorithms
                self.eps = self.eps + self.epsilon_increment if self.eps < self.epsilon_max else self.epsilon_max

        model.predict()

    def learn(self, step):
        if step < self.cache_size:
            idxs = np.random.choice(step, size=self.batch_dim)
        else:
            idxs = np.random.choice(self.cache_size, size=self.batch_dim)

        state_idxs = self.cache_state_idxs[idxs]
        actions = self.cache_actions[idxs]
        rewards = self.cache_reward[idxs]
        next_state_idxs = self.cache_next_state_idxs[idxs]

        q_nexts, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: next_state_idxs,
                self.s: next_state_idxs,
            })
        q_evals = self.sess.run(self.q_eval, {self.s: state_idxs})

        if self.ddqn:
            argmax = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_nexts[np.arange(self.batch_dim, dtype=int), argmax]
        else:
            selected_q_next = np.max(q_nexts, axis=1)

        q_targets = q_evals.copy()
        q_targets[np.arange(self.batch_dim, dtype=int), actions] = rewards + self.gamma * selected_q_next

        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.s: state_idxs,
                                           self.q_target: q_targets})
        if step % 300 == 0:
            self.sess.run(self.replace_target_op)
        if step % 1000 == 0:
            tqdm.write(str(loss), sys.stderr)

    def predict(self, start_idx=(0, 0)):
        state_idx = start_idx
        print('predict path: ' + str(state_idx), end='')
        i = 0
        while self.maze[state_idx] not in [1, 2] and i < 100:
            q_eval = self.sess.run(self.q_eval,
                                   feed_dict={
                                       self.s: np.array(list(state_idx)).reshape(
                                           (-1, self.n_features)) / self.maze.shape[0]})
            state = {k: q_eval[0][k] for k in range(self.n_actions)}
            if state_idx[0] == 0:  # no up
                state.pop(0)
            if state_idx[0] == self.maze.shape[0] - 1:  # no down
                state.pop(1)
            if state_idx[1] == 0:  # no left
                state.pop(2)
            if state_idx[1] == self.maze.shape[1] - 1:  # no right
                state.pop(3)
            action = self.choose(state, is_pred=1)
            idx = self.action_idx[action]
            next_state_idx = tuple((a + b for a, b in zip(state_idx, idx)))

            state_idx = next_state_idx
            print(' -> ' + str(state_idx), end='')
            maze = self.maze.copy()
            maze[state_idx] = 4
            self.show(maze, self.pixel, 'pred')
            i += 1

        if self.draw:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=50, blit=True,
                                            repeat_delay=1000, repeat=False)
            if save_fn:
                ani.save(self.save_fn, writer='imagemagick')
            plt.show()


if __name__ == '__main__':
    np.random.seed(2)

    # todo: if the maze is larger, the useless steps is much more, the network is worse
    a = np.zeros((6, 6), dtype=int)
    a[3, 3] = 1
    a[5, 5] = 2

    # save_fn = None
    save_fn = './img/DQN.gif'
    model = DQN(a, pixel=50, draw=False, save_fn=save_fn)
    model.train(1001)

"""
predict path: (0, 0) -> (1, 0) -> (1, 1) -> (1, 2) -> (2, 2) -> (3, 2) -> (3, 3)
"""
