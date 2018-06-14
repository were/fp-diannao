#!/usr/bin/env python2

import topi, tvm, autotvm, logging, time
from sa import SATuner

def dense(n, m, l):
    img = tvm.placeholder((n, m), dtype='float32')
    wei = tvm.placeholder((l, m), dtype='float32')
    res = topi.nn.dense(img, wei)

    cfg = autotvm.template.DispatchContext.current.query(None, None)

    cfg.add_flop(n * m * l * 2)

    sch = autotvm.cuda.schedule_dense(cfg, res)

    #sch = tvm.create_schedule(res.op)
    #autotvm.cuda.cuda_general_schedule(cfg, sch, res)

    return sch, [img, wei, res]


#dense(25088, 4096, 32)

logging.basicConfig(level=logging.INFO, filename='rnd-class-32.log')
#logging.basicConfig(level=logging.INFO)

tsk = autotvm.task.Task(dense, [25088, 4096, 32], {})
tgt = tvm.target.create('cuda -model=p4000')
tsk.init_space(tgt, None)

print(tsk.config_space)

tuner = autotvm.tuner.RandomTuner(tsk)
#tuner = autotvm.tuner.GATuner(tsk, pop_size=128, elite_num=4)
#tuner = autotvm.tuner.XGBTuner(tsk, tgt, 16, 4)
#tuner = SATuner(tsk, 128, 1024)

tuner.add_callback(autotvm.callback.SingleBestRecorder())

fmeature = autotvm.fleet.get_measure_batch(
    autotvm.fleet.create('titanx', timeout=10.0),
    tgt, 'llvm', repeat=5,
    retry_failures=None,
    replay_db=None,
    save=False,
    check_correctness=False)

logging.log(logging.INFO, str(len(tsk.config_space)))
start_time = time.time()
tuner.tune(fmeature, tgt, 800, 1)
logging.log(logging.INFO, str(time.time() - start_time))
