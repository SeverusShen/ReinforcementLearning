import numpy as np
import matplotlib.pyplot as plt


class RL:
    def __init__(self, maze, eps=0.8, gamma=0.9, lr=0.01, pixel=100, draw=False, save_fn=None):
        """
        maze: 0 means path, 1 means terminal, 2 means traps, 3 means obstacles, 4 means present state
        tables: (-1, 0) is down, (1, 0) is up, (0, -1) is left, (0, 1) is right
        """
        self.maze = maze
        self.eps = eps
        self.lr = lr
        self.gamma = gamma
        self.draw = draw
        self.save_fn = save_fn

        self.tables = None

        self.pixel = pixel
        img = np.zeros((maze.shape[0] * pixel, maze.shape[1] * pixel, 3), dtype=int) + 255
        terminal = np.where(maze == 1)
        for i in range(len(terminal[0])):
            img[terminal[0][i] * pixel:(terminal[0][i] + 1) * pixel,
            terminal[1][i] * pixel:(terminal[1][i] + 1) * pixel] = (200, 50, 50)
        traps = np.where(maze == 2)
        for i in range(len(traps[0])):
            img[traps[0][i] * pixel:(traps[0][i] + 1) * pixel, traps[1][i] * pixel:(traps[1][i] + 1) * pixel] = 0
        obstacles = np.where(maze == 3)
        for i in range(len(obstacles[0])):
            img[obstacles[0][i] * pixel:(obstacles[0][i] + 1) * pixel,
            obstacles[1][i] * pixel:(obstacles[1][i] + 1) * pixel] = 125
        self.img = img

        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ims = []

    def train(self, *args):
        raise NotImplementedError

    def predict(self, *args):
        raise NotImplementedError

    def choose(self, state, is_pred=0):
        if is_pred or np.random.uniform() < self.eps:  # choose best action
            action = max(state.items(), key=lambda x: x[1])[0]
        else:  # choose random action
            # todo: np.random.choice(state.keys()) don't work
            idx = np.random.randint(len(state.keys()))
            action = list(state.keys())[idx]
        return action

    def learn(self, *args):
        raise NotImplementedError

    def get_reward(self, s):
        if s == 1:
            reward = 1
        elif s == 2:
            reward = -1
        else:
            reward = 0
        return reward

    def get_table(self, maze):
        # init q_table
        tables = [[{} for _ in range(maze.shape[0])] for _ in range(maze.shape[1])]
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if i != 0 and maze[i - 1][j] != 3:  # have up
                    tables[i][j][(-1, 0)] = 0.
                if i != maze.shape[0] - 1 and maze[i + 1][j] != 3:  # have down
                    tables[i][j][(1, 0)] = 0.
                if j != 0 and maze[i][j - 1] != 3:  # have left
                    tables[i][j][(0, -1)] = 0.
                if j != maze.shape[1] - 1 and maze[i][j + 1] != 3:  # have right
                    tables[i][j][(0, 1)] = 0.
        return np.array(tables)

    def get_img(self, maze, pixel):
        img = np.zeros((maze.shape[0] * pixel, maze.shape[1] * pixel, 3), dtype=int) + 255

        terminal = np.where(maze == 1)
        for i in range(len(terminal[0])):
            img[terminal[0][i] * pixel:(terminal[0][i] + 1) * pixel,
            terminal[1][i] * pixel:(terminal[1][i] + 1) * pixel] = (200, 50, 50)

        traps = np.where(maze == 2)
        for i in range(len(traps[0])):
            img[traps[0][i] * pixel:(traps[0][i] + 1) * pixel, traps[1][i] * pixel:(traps[1][i] + 1) * pixel] = 0

        obstacles = np.where(maze == 3)
        for i in range(len(obstacles[0])):
            img[obstacles[0][i] * pixel:(obstacles[0][i] + 1) * pixel,
            obstacles[1][i] * pixel:(obstacles[1][i] + 1) * pixel] = 125

        path = np.where(maze == 4)
        for i in range(len(path[0])):
            img[path[0][i] * pixel:(path[0][i] + 1) * pixel,
            path[1][i] * pixel:(path[1][i] + 1) * pixel] = (50, 200, 50)

        return img

    def show(self, maze, pixel, itera):
        img = self.get_img(maze, pixel)
        self.ims.append([])
        # img[state_idx[0] * self.pixel:(state_idx[0] + 1) * self.pixel,
        # state_idx[1] * self.pixel:(state_idx[1] + 1) * self.pixel] = (50, 200, 50)
        self.ims[-1].append(self.ax.imshow(img))
        self.ims[-1].append(self.ax.text(pixel / 10, pixel / 2, 'itera: ' + str(itera)))
