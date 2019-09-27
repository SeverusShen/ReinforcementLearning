from reinforcement_learning.RL import RL
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Sarsa(RL):
    def train(self, itera=500, start_idx=(0, 0), lam=0.):
        self.tables = self.get_table(self.maze)
        self.lam = lam
        for i in tqdm(range(itera)):
            self.eligibility_trace = self.get_table(self.maze)
            state_idx = start_idx
            action = self.choose(self.tables[state_idx])
            if self.draw and i % (itera // 50) == 0:
                maze = self.maze.copy()
                maze[state_idx] = 4
                self.show(maze, self.pixel, i)
            while self.maze[state_idx] not in [1, 2]:
                next_state_idx = tuple((i + j for i, j in zip(state_idx, action)))
                next_action = self.choose(self.tables[next_state_idx])
                reward = self.get_reward(self.maze[next_state_idx])
                self.learn(state_idx, action, next_state_idx, next_action, reward)
                state_idx = next_state_idx
                action = next_action
                if self.draw and i % (itera // 10) == 0:
                    maze = self.maze.copy()
                    maze[state_idx] = 4
                    self.show(maze, self.pixel, i)
        self.predict()

    def predict(self, start_idx=(0, 0)):
        state_idx = start_idx
        print('predict path: ' + str(state_idx), end='')
        while self.maze[state_idx] not in [1, 2]:
            action = self.choose(self.tables[state_idx], is_pred=1)
            next_state_idx = tuple((i + j for i, j in zip(state_idx, action)))
            state_idx = next_state_idx
            print(' -> ' + str(state_idx), end='')
            maze = self.maze.copy()
            maze[state_idx] = 4
            self.show(maze, self.pixel, 'pred')

        if self.draw:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=50, blit=True,
                                            repeat_delay=1000, repeat=False)
            if save_fn:
                ani.save(self.save_fn, writer='imagemagick')
            plt.show()

    def learn(self, state_idx, action, next_state_idx, next_action, reward):
        next_state = self.tables[next_state_idx]
        value = next_state[next_action]
        if self.maze[next_state_idx] in [1, 2]:
            q_target = reward
        else:
            q_target = reward + self.gamma * value
        error = q_target - self.tables[state_idx][action]

        # self.tables[state_idx][action] += self.lr * error

        self.eligibility_trace[state_idx][action] += 1

        # train speed is much slower
        # Q update
        for i in range(self.tables.shape[0]):
            for j in range(self.tables.shape[1]):
                for k in self.tables[i][j]:
                    # print(self.eligibility_trace[i][j][k])
                    self.tables[i][j][k] += self.lr * error * self.eligibility_trace[i][j][k]
                    # print(self.tables[i][j][k])

        # decay eligibility trace after update
        for i in range(self.eligibility_trace.shape[0]):
            for j in range(self.eligibility_trace.shape[1]):
                for k in self.eligibility_trace[i][j]:
                    self.eligibility_trace[i][j][k] *= self.gamma * self.lam


if __name__ == '__main__':
    import numpy as np

    np.random.seed(1)
    a = np.zeros((11, 11), dtype=int)
    a[8, 8] = 1
    a[2, 2] = a[2, 8] = a[8, 2] = 2
    a[5, 2:9] = a[2:9, 5] = 3
    save_fn = None
    # save_fn = './img/Sarsa.gif'
    model = Sarsa(a, pixel=50, draw=True, save_fn=save_fn)
    model.train(itera=2501)

"""
predict path: (0, 0) -> (1, 0) -> (2, 0) -> (3, 0) -> (4, 0) -> (5, 0) -> (5, 1) -> (6, 1) -> (6, 2) -> (6, 3) 
-> (7, 3) -> (7, 4) -> (8, 4) -> (9, 4) -> (9, 5) -> (9, 6) -> (9, 7) -> (8, 7) -> (8, 8)
"""
