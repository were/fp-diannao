#!/usr/bin/env python2

import topi, tvm, autotvm, time, numpy, logging
from autotvm import MeasureResult, MeasureErrorNo

def conv(iw, ih, fw, fh, fi, fo, batch, dtype):
    img = tvm.placeholder((batch, fi, iw, ih), dtype=dtype, name='img')
    fil = tvm.placeholder((fi, fo, fw, fh), dtype=dtype, name='fil')

    conv= topi.nn.conv2d_nchw(img, fil, (1, 1), 'VALID')
    sch = tvm.create_schedule(conv.op)

    _, _, pad, _ = [k for k,v in sch.stage_map.items()]
    #print(sch[conv].op.axis)
    n, f, y, x = sch[conv].op.axis
    rc, ry, rx = sch[conv].op.reduce_axis

    cfg = autotvm.template.DispatchContext.current.query(None, None)

    cfg.add_flop(iw * ih * fw * fh * fi * fo * batch)

    cfg.split('tile_n', cfg.axis(n), policy='all', num_split=2)
    cfg.split('tile_f', cfg.axis(f), policy='all', num_split=2)
    cfg.split('tile_y', cfg.axis(y), policy='all', num_split=2)
    #cfg.split('tile_c', cfg.axis(rc), policy='all', num_split=2)

    no, ni = cfg['tile_n'].apply(sch, conv, n)
    fo, fi = cfg['tile_f'].apply(sch, conv, f)
    yo, yi = cfg['tile_y'].apply(sch, conv, y)
    xo, xi = sch[conv].split(x, 4)
    #co, ci = cfg['tile_c'].apply(sch, conv, rc)
    sch[conv].vectorize(xi)

    axis = [no, ni, fo, fi, yo, yi, rc, ry, rx, xo]
    cfg_axis = [cfg.axis(i) for i in axis]
    cfg_axis = [i for i in axis]
    cfg.reorder('reorder', axis, policy='all')
    cfg['reorder'].apply(sch, conv, axis)
    #sch[conv].reorder(no, ni, fo, fi, yo, yi, rc, ry, rx, xo)

    sch[pad].compute_at(sch[conv], xo)

    return sch, [img, fil, conv]

def measure_batch(measure_inputs):
    results = []
    ctx = tvm.cpu()
    for inp in measure_inputs:
        target, tsk, cfg = inp
        try:
            print(cfg)
            # instantiate tuning
            with target:
                s, arg_bufs = tsk.instantiate(cfg)

            #print(tvm.lower(s, arg_bufs, simple_mode=True))

            # build function
            func = tvm.build(s, arg_bufs)

            # measure time
            arg_bufs = [tvm.nd.array(numpy.random.randn(*[d.value for d in x.shape]).astype(x.dtype), ctx) for x in arg_bufs]
            time_f = func.time_evaluator(func.entry_name, ctx, number=1, repeat=5)
            prof = time_f(*arg_bufs)

            #print('mean: %f ms' % prof.mean)

            res = MeasureResult(tuple(prof.results), prof.mean, MeasureErrorNo.NO_ERROR, time.time())
        except Exception as e:
            # if there is any error
            res = MeasureResult((e,), 1e9, MeasureErrorNo.INSTANTIATION_ERROR, time.time())
        results.append(res)

    return results

logging.basicConfig(level=logging.INFO, filename='random.log')
tsk = autotvm.task.Task(conv, [226, 226, 3, 3, 64, 64, 1, 'float32'], {})
tgt = tvm.target.create('llvm')
tsk.init_space(tgt, None)
#print(tsk.config_space)

#tuner = autotvm.tuner.GridSearchTuner(tsk)
tuner = autotvm.tuner.RandomTuner(tsk)

tuner.add_callback(autotvm.callback.SingleBestRecorder())

logging.log(logging.INFO, str(len(tsk.config_space)))
start_time = time.time()
tuner.tune(measure_batch, tgt, 1000, 1)
logging.log(logging.INFO, str(time.time() - start_time))
