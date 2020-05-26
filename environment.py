import numpy as np

class Environment:

    state = []
    goal = []
    boundary = []

    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    char_map = {0: 'nowhere', 1: 'right', 2: 'left', 3: 'down', 4: 'up'}

    def __init__(self, x, y, initial, goal, sand_penalization=-1):


        # Dictionary of conversion
        self.value_map =  {'Goal': 5, 'Agent': 3, 'Path': 2, 'Path\n(In Sand)': 1, 'Terrain': 0, 'Sand': -1, 'Obstacle': -2}
        self.value2char = {self.value_map[key]: key for key in self.value_map}

        # Init enviroment
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.start = np.asarray(initial)
        self.goal = goal

        # Init obstacle
        self.obstacle = 0
        self.sand = 0
        self.sand_penalization = sand_penalization

        # Init matrix
        self.matrix = np.zeros((x, y))
        self.matrix[goal[0], goal[1]] = self.value_map['Goal']
        self.matrix[initial[0], initial[1]] = self.value_map['Agent']

    # Implement print
    def __str__(self):
        return str(self.matrix)

    # Init Obstacle
    def create_obstacle(self, obstacle):
        for pos in obstacle:
            if (pos == self.state).all():
                continue
            x, y = pos[0], pos[1]
            self.matrix[x, y] = self.value_map['Obstacle']
        self.obstacle = obstacle

    # Init sand
    def create_sand(self, sand):
        for pos in sand:
            if (pos == self.state).all() or (self.matrix[pos[0], pos[1]] == self.value_map['Obstacle']) :
                continue
            x, y = pos[0], pos[1]
            self.matrix[x, y] = self.value_map['Sand']
        self.sand = sand

    # The agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action, verbose=False):

        # Get the action
        reward = 0
        movement = self.action_map[action]
        next_state = self.state + np.asarray(movement)

        # Mark the path
        if not ((self.state == self.goal).all()):
            if list(self.state) in self.sand:
                self.matrix[self.state[0], self.state[1]] = self.value_map['Path\n(In Sand)']
            else:
                self.matrix[self.state[0], self.state[1]] = self.value_map['Path']

        # Check bound and obstacle
        bound = self.check_boundaries(next_state)
        obs = list(next_state) in self.obstacle
        sand = list(next_state) in self.sand

        # Print info
        if verbose:
            print('Movement is: {}'.format(char_map[action]))
            print('Next possible state is: {}'.format(next_state))

            print('Is out of boundaries: {}'.format(bound))
            print('Is an obstacle: {}'.format(obs))

        # Check the reaching of the goal
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1

        # Check if we are on a sand
        if sand:
            reward = self.sand_penalization

        # Check if we went out or on an obstacle
        if bound or obs:
            reward = -1
        else:
            self.state = next_state

        # Mark new position
        self.matrix[self.state[0], self.state[1]] = self.value_map['Agent']

        # Return state and reward
        return [self.state, reward]

    # Map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        return out > 0

if __name__ == '__main__':
    import dill
    import agent
    import matrixlib as matpl
    import sys
    print(sys.executable)

    # Init variables
    obstacle= [[0, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [0, 7], [3, 4], [5, 4], [1, 7], [8, 4], [9, 9], [4, 4], [6, 4], [7, 4]]
    sand = [[5, 0], [5, 1], [5, 2], [3, 3], [3, 5], [3, 6], [3, 7], [2, 7]]
    goal = [0, 3]
    initial = state = [np.random.randint(0, 10), np.random.randint(0, 10)]
    print('Starting from: {}'.format(initial))
    print('Goal at: {}'.format(goal))

    # Create environment
    env = Environment(10, 10, initial, goal)
    env.create_obstacle(obstacle)
    env.create_sand(sand)

    # Show initial status
    im = matpl.plot(env.matrix, env, figsize=(6, 6), reduct=True)
    matpl.add_patches(im, env)

    # Retrieve agent
    with open('learners\\sarsa_low.obj', 'rb') as agent_file:
        agent = dill.load(agent_file)

    episodes = 5000
    episode_length = 50
    epsilon = np.linspace(0.1, 0.001, 5000)

    # Perform actions
    actions = []
    reward = 0
    # run episode
    for step in range(0, episode_length):
        # find state index
        state_index = state[0] * 10 + state[1]

        #print('State index:', state_index)
        # choose an action
        action = agent.select_action(state_index, epsilon[-1])
        #print('Action:', action)
        # the agent moves in the environment
        result = env.move(action, verbose=False)
        #print('Result:', result)
        # Q-learning update
        next_index = result[0][0] * 10 + result[0][1]
        reward += result[1]
        actions.append(env.matrix.copy())
        #print('----------')
        state = result[0]
    # Show final status
    im = matpl.plot(env.matrix, env, figsize=(6, 6), reduct=True)
    matpl.add_patches(im, env)
