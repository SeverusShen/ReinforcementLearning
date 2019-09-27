from reinforcement_learning.RL import RL
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Q_learning(RL):
    def train(self, itera=500, start_idx=(0, 0)):
        self.tables = self.get_table(self.maze)
        for i in tqdm(range(itera)):
            state_idx = start_idx
            if self.draw and i % (itera // 50) == 0:
                maze = self.maze.copy()
                maze[state_idx] = 4
                self.show(maze, self.pixel, i)
            while self.maze[state_idx] not in [1, 2]:
                action = self.choose(self.tables[state_idx])
                next_state_idx = tuple((a + b for a, b in zip(state_idx, action)))
                reward = self.get_reward(self.maze[next_state_idx])
                self.learn(state_idx, action, next_state_idx, reward)
                state_idx = next_state_idx
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

    def learn(self, state_idx, action, next_state_idx, reward):
        next_state = self.tables[next_state_idx]
        value = max(next_state.items(), key=lambda x: x[1])[1]
        reward += self.gamma * value - self.tables[state_idx][action]
        self.tables[state_idx][action] += self.lr * reward


if __name__ == '__main__':
    import numpy as np

    a = np.zeros((11, 11), dtype=int)
    a[8, 8] = 1
    a[2, 2] = a[2, 8] = a[8, 2] = 2
    a[5, 2:9] = a[2:9, 5] = 3
    save_fn = None
    # save_fn = './img/Q_learning.gif'
    model = Q_learning(a, pixel=50, draw=True, save_fn=save_fn)
    model.train(itera=2501)

"""
predict path: (0, 0) -> (1, 0) -> (2, 0) -> (3, 0) -> (3, 1) -> (4, 1) -> (5, 1) -> (6, 1) -> (6, 2) -> (7, 2) -> (7, 3) 
-> (8, 3) -> (9, 3) -> (9, 4) -> (9, 5) -> (9, 6) -> (9, 7) -> (8, 7) -> (8, 8)
"""
