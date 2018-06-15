#!/usr/bin/env python2

import topi, tvm, autotvm, logging, time
from sa import SATuner
from ga import GATuner

@autotvm.simple_template
def matvec(n, m, l):
    wei = tvm.placeholder((n, m), dtype='float32')
    data= tvm.placeholder((l, m), dtype='float32')
    res = topi.nn.dense(img, wei)

    cfg = autotvm.template.DispatchContext.current.query(None, None)

    s = tvm.create_schedule(res.op)

    if not tvm.gpu(0).exist:
        raise ValueError('shit!')

    n, k = get_const_tuple(data.shape)
    m, _ = get_const_tuple(wei.shape)
    cfg.add_flop(2 * n * l * m)

    output = den
    OL = s.cache_write(den, 'local')

    # create cache stage
    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(weight, 'shared', [OL])

    # bind
    y, x = s[output].op.axis
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
    scope, y = s[output].split(y, nparts=1)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    s[output].bind(by, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(scope, by, bx, vy, vx, ty, tx, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile and bind reduction axes
    y, x = s[OL].op.axis
    r, = s[OL].op.reduce_axis
    cfg.define_split("tile_r", cfg.axis(r), num_outputs=3)
    ro, rm, ri = cfg['tile_r'].apply(s, OL, r)
    s[OL].reorder(ro, rm, ri, y, x)

    s[AA].compute_at(s[OL], ro)
    s[WW].compute_at(s[OL], rm)
    # s[AL].compute_at(s[OL], rxm)
    # s[WL].compute_at(s[OL], rxm)

    for load in [AA, WW]:
        fused = s[load].fuse(*list(s[load].op.axis))
        fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    cfg.other_option("auto_unroll_max_step", [0, 512, 1500])
    cfg.other_option("unroll_explicit", [0, 1])
    s[output].pragma(scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    return s, [img, wei, res]


logging.basicConfig(level=logging.INFO, filename='ga-class-1.log')
#logging.basicConfig(level=logging.INFO)

tsk = autotvm.task.create_task('matvec', [25088, 1, 4096], 'cuda', 'llvm')
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
