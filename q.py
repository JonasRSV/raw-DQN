import autograd.numpy as np
import gym
from autograd import grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Q:
    def __init__(self,
                 arch,
                 loss,
                 hidden_activation=sigmoid,
                 output_activation=sigmoid,
                 epsilon=0.1):

        self.arch = arch
        self.loss = loss
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.epsilon = epsilon

        self.replay = []

        self.action_space = self.arch[-1][0].shape[-1]

        def objective(params, X, Y):
            for W, b in params[:-1]:
                X = self.hidden_activation(X @ W + b)
            out = self.output_activation(X @ params[-1][0] + params[-1][1])

            return self.loss(out, Y)

        self.gradient = grad(objective)

    def predict(self, x):
        for W, b in self.arch[:-1]:
            x = self.hidden_activation(x @ W + b)
        return self.output_activation(x @ self.arch[-1][0] + self.arch[-1][1])

    def act(self, x, deterministic=True):
        probabilities = self.predict(x).reshape(-1)

        if deterministic:
            return np.argmax(probabilities)
        else:
            return np.random.choice([0, 1])

    def observe(self, game):
        # Is a list of ([s0], [a0], [s1], [r], [d])
        s0, a0, s1, r, d = game

        self.replay.extend(zip(s0, a0, s1, r, d))

    def unwrap_batch(self, batch):
        s0s, a0s, s1s, rs, ds = [], [], [], [], []

        for s0, a0, s1, r, d in batch:
            s0s.append(s0)
            a0s.append(a0)
            s1s.append(s1)
            rs.append(r)
            ds.append(d)

        s0s, a0s, s1s, rs, ds = np.array(s0s), np.array(a0s),\
                np.array(s1s), np.array(rs), np.array(ds)

        return s0s, a0s, s1s, rs, ds

    def train(self, value_iters, grad_iters):
        """Use multiple value iterations to learn the MDP."""

        # To make loss printing look nice
        losses = 0
        smooth_loss = 0

        indexes = np.arange(0, len(self.replay))
        for vi in range(value_iters):
            vi_batch = [self.replay[i] for i in np.random.choice(indexes, 256)]

            s0s, a0s, s1s, rs, ds = self.unwrap_batch(vi_batch)

            s0_utility = self.predict(s0s)
            s1_utility = self.predict(s1s)

            target = []
            # This is the value iteration step, where we learn what the utility should be
            for s0u, s1u, a, r, d in zip(s0_utility, s1_utility, a0s, rs, ds):
                s0u[a] = r + (self.epsilon * np.max(s1u)) * (1 - float(d))
                target.append(s0u)

            target = np.array(target)
            grad_indexes = np.arange(0, len(target))
            # This is the update representation step. We try to learn the mapping
            # state -> utility using a neural network
            for i in range(grad_iters):
                batch = np.random.choice(grad_indexes, 32)

                g = self.gradient(self.arch, s0s[batch], target[batch])

                for j, _ in enumerate(self.arch):
                    self.arch[j][0] -= 0.01 * g[j][0]
                    self.arch[j][1] -= 0.01 * g[j][1]

            if vi % 10 == 0:
                loss = self.loss(self.predict(s0s), target)
                smooth_loss = (smooth_loss * losses + loss) / (losses + 1)
                losses += 1

                print("vi", vi, "loss", smooth_loss, end="               \r")

        return loss


def loss(pred_u, target_u):
    return np.mean(np.square(pred_u - target_u))


def demo_q(train_iters, demo_iters):
    hidden = 64
    scale = 0.5
    arch = [[
        np.random.normal(scale=scale, size=(4, hidden)),
        np.random.normal(scale=scale, size=hidden)
    ],
            [
                np.random.normal(scale=scale, size=(hidden, hidden)),
                np.random.normal(scale=scale, size=hidden)
            ],
            [
                np.random.normal(scale=scale, size=(hidden, 2)),
                np.random.normal(scale=scale, size=2)
            ]]

    q = Q(arch=arch,
          loss=loss,
          epsilon=0.95,
          hidden_activation=sigmoid,
          output_activation=lambda x: x)

    env = gym.make('CartPole-v0')

    for e in range(train_iters):
        ss0 = []
        aa0 = []
        ss1 = []
        rr = []
        dd = []

        s0 = env.reset()
        done = False
        while not done:
            a0 = q.act(s0.reshape(1, -1), deterministic=False)
            s1, r, done, _ = env.step(a0)  # take a random action

            ss0.append(s0)
            aa0.append(a0)
            ss1.append(s1)
            rr.append(r)
            dd.append(done)

            s0 = s1

        q.observe((ss0, aa0, ss1, rr, dd))

    l = q.train(1000, 10)

    for _ in range(demo_iters):
        s0 = env.reset()
        done = False
        rew = 0
        while not done:
            env.render()
            a0 = q.act(s0.reshape(1, -1))
            #print(a0)
            s1, r, done, _ = env.step(a0)  # take a random action
            s0 = s1
            rew += r

        print(rew)

    env.close()


demo_q(200, 40)
