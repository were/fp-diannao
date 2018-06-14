import autotvm
import random
import numpy as np
import logging
from autotvm.tuner.tuner import Tuner

class SATuner(Tuner):
    def __init__(self, task, pool, total, shrink=0.9, T=None, target=None):
        super(SATuner, self).__init__(task)

        self.space = task.config_space
        self.pool_size = pool
        self.total = total
        self.shrink = shrink

        vals = self.space.space_map.values()

        self.dims = [len(i) for i in vals] + [1]
        self.n_dims = len(vals)
        if T is None:
            self.originT = self.T = np.sqrt(sum(i * i for i in self.dims)) / pool
        else:
            self.originT = self.T = T
        self.N = reduce(int.__mul__, self.dims)

        self.visited = set([])

        self.pool = []
        self.exec_pool = []
        for i in range(pool):
            temp = random.randint(0, self.N - 1)
            while temp in self.visited:
                temp = random.randint(0, self.N - 1)
            self.pool.append((temp, 0))
            self.exec_pool.append(temp)
            self.visited.add(temp)

            vec = self._i_to_vec(temp)
            #print temp
            #print vec
            #print self._vec_to_i(vec)

        self.cur = 0
        self.better = False


    def next_batch(self, batch_size):
        res = []
        for i in range(self.cur, min(self.pool_size, self.cur + batch_size)):
            res.append(self.space.get(self.exec_pool[i]))
            self.cur += 1
        return res

    def has_next(self):
        return len(self.visited) - (self.pool_size - self.cur) < self.total and self.T > 1.0

    def update(self, inputs, results):
        n = len(inputs)
        assert n == len(results)
        for i in range(self.cur - n, self.cur):
            _i = i - self.cur + n
            inp = inputs[_i]
            res = results[_i]
            sol, old_score = self.pool[i]
            if res.error_no == 0:
                new_score = inp.task.flop / np.mean(res.runs) / 2.5e12
            else:
                new_score = 0.0
            if new_score > old_score:
                logging.log(logging.INFO, '%f is better than %f' % (new_score, old_score))
                self.pool[i] = (self.exec_pool[i], new_score)
                self.better = True
            elif random.random() < np.exp(-(old_score - new_score) * self.originT / self.T):
                per = np.exp(-(old_score - new_score) * self.originT / self.T)
                logging.log(logging.INFO, '%.2f%% accept a bad score %f (%f)' % (per * 100.0, new_score, old_score))
                self.pool[i] = (self.exec_pool[i], new_score)

        if self.cur >= self.pool_size:
            if self.better:
                self.T *= self.shrink
                logging.log(logging.INFO, 'Current Temperature: %f' % self.T)
                self.better = False

            self.cur = 0
            for i in range(self.pool_size):
                sol, _ = self.pool[i]

                def _explore(sol):
                    vec = self._i_to_vec(sol)
                    T = self.T
                    for j in range(self.n_dims - 1):
                        angle = (random.random() * np.pi * 2)
                        vec[j] = (vec[j] + int(self.T * np.cos(angle))) % self.dims[j]
                    T *= np.sin(angle)
                    vec[-1] = (vec[-1] + int(T)) % self.dims[-1]
                    return self._vec_to_i(vec)

                temp = _explore(sol)
                while temp in self.visited:
                    temp = _explore(temp)
                self.exec_pool[i] = temp
            

    def _i_to_vec(self, num):
        res = []
        for i in range(self.n_dims, 0, -1):
            res.append(num % self.dims[i])
            num /= self.dims[i]
        res.reverse()
        return res

    def _vec_to_i(self, vec):
        res = 0
        for i in range(1, self.n_dims):
            res = res * self.dims[i] + vec[i-1]
        return res

