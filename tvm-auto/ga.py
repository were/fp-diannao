import autotvm
import random
import numpy as np
import logging
from autotvm.tuner.tuner import Tuner

class GATuner(Tuner):
    def __init__(self, task, n_pool, n_elites, mutant=0.1):
        super(GATuner, self).__init__(task)

        self.space    = task.config_space
        self.n_pool   = n_pool
        self.n_elites = n_elites
        self.mutant   = mutant

        vals = self.space.space_map.values()

        self.dims = [len(i) for i in vals] + [1]
        self.n_dims = len(vals)

        self.N = reduce(int.__mul__, self.dims)

        self.visited = set([])

        self.pool = []
        self.elites = []
        self.exec_pool = []
        for i in range(n_pool):
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
        for i in range(self.cur, min(self.n_pool, self.cur + batch_size)):
            res.append(self.space.get(self.exec_pool[i]))
            self.cur += 1
        return res

    def has_next(self):
        return len(self.visited) - (self.n_pool - self.cur) < len(self.space)

    def update(self, inputs, results):
        n = len(inputs)
        assert n == len(results)
        for i in range(self.cur - n, self.cur):
            _i = i - self.cur + n
            inp = inputs[_i]
            res = results[_i]
            if res.error_no == 0:
                new_score = inp.task.flop / np.mean(res.costs) / 2.5e12
            else:
                new_score = 0.0
            self.pool.append((self.exec_pool[i], new_score))

        if self.cur >= self.n_pool:
            self.pool += self.elites

            def _cmp_tpl(a, b):
                _, key_a = a
                _, key_b = b
                if key_b - key_a < -1e-6:
                    return -1
                elif key_b - key_a > 1e-6:
                    return 1
                return 0

            self.pool.sort(_cmp_tpl)
            self.pool = self.pool[:self.n_pool]
            self.elites = self.pool[:self.n_elites]

            self.cur = 0
            self.exec_pool = []
            for i in range(self.n_pool):

                def _hybrid_and_mutant(_a, _b, x):
                    #hybrid
                    a = self._i_to_vec(_a)
                    b = self._i_to_vec(_b)
                    res = a[:x] + b[x:]
                    #mutant
                    for j in range(self.n_dims):
                        if random.random() < self.mutant:
                            res[j] = random.randint(0, self.dims[j] - 1)
                    return self._vec_to_i(res)

                x = self.pool[random.randint(0, self.n_pool - 1)][0]
                y = self.pool[random.randint(0, self.n_pool - 1)][0]
                z = random.randint(0, self.n_dims - 1)
                temp = _hybrid_and_mutant(x, y, z)

                while temp in self.visited:
                    x = self.pool[random.randint(0, self.n_pool - 1)][0]
                    y = self.pool[random.randint(0, self.n_pool - 1)][0]
                    z = random.randint(0, self.n_dims - 1)
                    temp = _hybrid_and_mutant(x, y, z)

                self.exec_pool.append(temp)
                self.visited.add(temp)


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

