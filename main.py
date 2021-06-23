import unittest
from functools import lru_cache
from itertools import zip_longest
from enum import IntEnum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.animation import MovieWriter, FFMpegWriter
import seaborn as sns


class Strategy(IntEnum):
    soft = 0
    bin = 1


def arange(lower, upper):
    def _range(f):
        f.range = (lower, upper)
        return f
    return _range


@arange(-5.0, 5.0)
def ackley(data):
    assert data.shape[1] == 2
    x = data[:, 0]
    y = data[:, 1]
    first_term = -20 * np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
    second_term = -np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e + 20
    return second_term + first_term


@arange(-5.12, 5.12)
def rastrigin(data):
    n = data.shape[1]
    sum = 10 * n
    return 10 * n + np.sum(data ** 2 - 10 * np.cos(2 * np.pi * data), axis=1)


@arange(-100.0, 100.0)
def schaffer(data):
    assert data.shape[1] == 2
    x = data[:, 0]
    y = data[:, 1]
    return 0.5 + ((np.sin(x**2-y**2)**2 - 0.5) / (1 + 0.001 * (x**2+y**2))**2)


@arange(-100.0, 100.0)
def rosenbrock(data):
    return np.sum(100 * (data[:, 1:] - data[:, :-1]**2) ** 2 + (1 - data[:, :-1])**2, axis=1)


@lru_cache(maxsize=1)
def _center_function(population_size):
    centers = np.arange(0, population_size, dtype=np.float32)
    centers = centers / (population_size - 1)
    centers -= 0.5
    centers *= 2.0
    return centers


def _compute_ranks(rewards):
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks


def rank_transformation(rewards):
    ranks = _compute_ranks(rewards)
    values = _center_function(rewards.size)
    return values[ranks]


def evaluate_population(population, func=ackley):
    return -func(population)


class TestOptimum(unittest.TestCase):
    def test_ackley(self):
        self.assertEqual(ackley(np.array([[0, 0]])), 0)

    def test_rastrigin(self):
        self.assertEqual(rastrigin(np.array([[0, 0]])), 0)
        self.assertEqual(rastrigin(np.array([[0, 0, 0]])), 0)

    def test_schaffer(self):
        self.assertEqual(schaffer(np.array([[0, 0]])), 0)

    def test_rosenbrock(self):
        self.assertEqual(rosenbrock(np.array([[1, 1]])), 0)
        self.assertEqual(rosenbrock(np.array([[1, 1, 1]])), 0)
        self.assertEqual(rosenbrock(np.ones((1, 100))), 0)


if __name__ == '__main__':
    unittest.main(exit=False)
    VISUALIZE = False
    PLOT = True
    MUTATION_FACTOR = 0.7
    CROSSOVER_PROBABILITY = [0.7, 0.5, 0.05]
    POLYAK = [0.995, 0.95, 0.5]
    POPULATION_SIZE = 128
    DIMENSION = 2
    # DIMENSION = 256

    FUNC = ackley
    # FUNC = rastrigin
    # FUNC = schaffer
    # FUNC = rosenbrock

    if VISUALIZE:
        assert DIMENSION == 2
        X = np.linspace(*FUNC.range, 100)     
        Y = np.linspace(*FUNC.range, 100)     
        X, Y = np.meshgrid(X, Y) 
        Z = FUNC(X, Y)
        fig3d, ax3d = plt.subplots(subplot_kw={"projection": "3d"})
        moviewriter = FFMpegWriter()
        moviewriter.setup(fig3d, 'animation.mp4', dpi=100)

    STEPS = 100
    SEEDS = [7235, 4050, 5935, 2919, 2740, 7210, 4012, 5936, 2920, 2741]
    data = []
    binomial = list(zip_longest([], CROSSOVER_PROBABILITY, fillvalue=Strategy.bin))
    soft = list(zip_longest([], POLYAK, fillvalue=Strategy.soft))
    for strategy, cr in binomial + soft:
        for seed in SEEDS:
            np.random.seed(seed)
            population = np.random.uniform(*FUNC.range, (POPULATION_SIZE, DIMENSION))

            # population = np.random.uniform(1.0, FUNC.range[1], (POPULATION_SIZE, DIMENSION))

            rewards = evaluate_population(population, FUNC)
            res = {'generation': 0, 'reward': np.max(rewards),
                   'strategy': strategy.name, 'f': MUTATION_FACTOR, 'cr': cr,
                   'seed': seed}
            data.append(res)

            for i in range(STEPS):
                candidate_population = []
                for j in range(POPULATION_SIZE):
                    best_idx = np.argmax(rewards)
                    idxs = np.random.choice(np.delete(np.arange(POPULATION_SIZE), j), 2, replace=False)
                    sub_rewards = rank_transformation(rewards)[idxs]
                    distances = population[idxs] - population[j]
                    diff = sub_rewards @ distances
                    mutation_vector = population[best_idx] + MUTATION_FACTOR * diff

                    if strategy == Strategy.bin:
                        cross = np.random.rand(DIMENSION) <= cr
                        new_candidate = population[j].copy()
                        new_candidate[cross] = mutation_vector[cross]
                    elif strategy == Strategy.soft:
                        new_candidate = population[j].copy()
                        new_candidate = cr * new_candidate + (1 - cr) * mutation_vector
                    else:
                        raise NotImplementedError
                    candidate_population.append(new_candidate)
                        
                candidate_population = np.array(candidate_population)
                candidate_rewards = evaluate_population(candidate_population, FUNC)
                condition = candidate_rewards > rewards
                population[condition] = candidate_population[condition]
                rewards[condition] = candidate_rewards[condition]
                res = {'generation': i+1, 'reward': np.max(rewards),
                       'strategy': strategy.name, 'f': MUTATION_FACTOR, 'cr': cr,
                       'seed': seed}
                data.append(res)

                if VISUALIZE:
                    ax3d.cla()
                    ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.nipy_spectral,
                                      linewidth=0.08, antialiased=True)    
                    p = np.array(population)
                    ax3d.plot(p[:, 0], p[:, 1], 'ro') 
                    plt.draw()
                    moviewriter.grab_frame()
                    plt.pause(0.001)
    if VISUALIZE:
        moviewriter.finish()

    if PLOT:
        data = pd.DataFrame(data)
        data['settings'] = data['strategy']
        data.loc[data['settings'] == 'bin', 'settings'] += ' cr: ' + data['cr'].apply(str)
        data.loc[data['settings'] == 'soft', 'settings'] += ' polyak: ' + data['cr'].apply(str)

        fig, ax = plt.subplots()
        ax = sns.lineplot(ax=ax, x='generation', y="reward", data=data, ci='sd', hue='settings',
                          estimator=getattr(np, 'mean'), linewidth=0.8)
        ax.set(xlabel='Generation', ylabel='Reward')
        ax.xaxis.set_label_position('top')
        ax.title.set_text(f'Func: {FUNC.__name__} D: {DIMENSION} F: {MUTATION_FACTOR} Population Size: {POPULATION_SIZE}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)).set_draggable(True)
        plt.tight_layout(pad=0.5)
        plt.savefig(f'plot_{FUNC.__name__}_{DIMENSION}_{MUTATION_FACTOR}_{POPULATION_SIZE}.png')


