#!/usr/bin/env python2

import topi, tvm, autotvm, logging, time
from sa import SATuner
from ga import GATuner

@autotvm.simple_template
def matvec(n, m, l):
    img = tvm.placeholder((n, m), dtype='float32')
    wei = tvm.placeholder((l, m), dtype='float32')
    res = topi.nn.dense(img, wei)

    cfg = autotvm.template.DispatchContext.current.query(None, None)


    sch = autotvm.cuda.schedule_dense(cfg, res)

    if not tvm.gpu(0).exist:
        raise ValueError('shit!')

    #cfg.add_flop(n * m * l * 2)
    #sch = tvm.create_schedule(res.op)
    #autotvm.cuda.cuda_general_schedule(cfg, sch, res)

    return sch, [img, wei, res]


logging.basicConfig(level=logging.INFO, filename='ga-class-32.log')
#logging.basicConfig(level=logging.INFO)

tsk = autotvm.task.create_task('matvec', [25088, 32, 4096], 'cuda', 'llvm')
tsk.init_space('cuda', 'llvm')
print(tsk.config_space)

#tuner = autotvm.tuner.RandomTuner(tsk)
tuner = GATuner(tsk, n_pool=16, n_elites=4)
#tuner = autotvm.tuner.XGBTuner(tsk, tgt, 16, 4)
#tuner = SATuner(tsk, 128, 1024)

logging.log(logging.INFO, str(len(tsk.config_space)))
start_time = time.time()
tuner.tune(n_trial=800, 
           measure_option=autotvm.measure_option(mode='local-nofork',
                                                 timeout=20.0,
                                                 repeat=5,
                                                 replay_db=None,
                                                 save_to_replay_db=False,
                                                 check_correctness=False))
logging.log(logging.INFO, str(time.time() - start_time))
