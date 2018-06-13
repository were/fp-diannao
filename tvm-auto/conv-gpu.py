#!/usr/bin/env python2

import topi, tvm, autotvm, time, numpy, logging, operator
from autotvm import MeasureResult, MeasureErrorNo
from sa import SATuner

def conv(iw, ih, fw, fh, fi, fo, batch, dtype):
    img = tvm.placeholder((batch, fi, iw, ih), dtype=dtype, name='img')
    fil = tvm.placeholder((fi, fo, fw, fh), dtype=dtype, name='fil')

    conv= topi.nn.conv2d_nchw(img, fil, (1, 1), 'VALID')

    sch = tvm.create_schedule(conv.op)

    temp = conv.op.input_tensors[0]
    sch[temp].compute_inline()

    #share = sch.cache_read(conv.op.input_tensors[1], 'shared', [conv])
    #local = sch.cache_read(share, 'local', [conv])

    cfg = autotvm.template.DispatchContext.current.query(None, None)

    cfg.add_flop(iw * ih * fw * fh * fi * fo * batch * 2)

    #nr_axes  = [cfg.axis(i) for i in conv.op.axis]
    #nr_splits = [cfg.split('tile_' + i.name, i, policy='all', num_split=2) for i in nr_axes]
    #ord_axes = cfg.reorder('reorder', reduce(list.__add__, nr_splits), policy='interleave', spatial=nr_splits, reduce=[])
    #cfg.annotate('binds', ord_axes[:sum([len(axis) - 1 for axis in nr_splits])], policy='bind_gpu_virtual')

    #red_axes  = [cfg.axis(i) for i in conv.op.reduce_axis]
    #red_splits= [cfg.split('tile_red_' + i.name, i, policy='all', num_split=2) for i in red_axes]

    #nr_splits = [cfg['tile_' + i.var.name].apply(sch, conv, i) for i in sch[conv].op.axis]
    ##red_splits= [cfg['tile_red_' + i.var.name].apply(sch, conv, i) for i in sch[conv].op.reduce_axis]
    #order     = cfg['reorder'].apply(sch, conv, reduce(list.__add__, nr_splits))
    ##print(list(map(len, nr_splits)))
    #binds = cfg['binds'].apply(sch, conv, order[:sum([len(axis) - 1 for axis in nr_splits])])

    #print tvm.lower(sch, [img, fil, conv], simple_mode=True)
    autotvm.cuda.cuda_general_schedule(cfg, sch, conv)

    return sch, [img, fil, conv]

logging.basicConfig(level=logging.INFO, filename='sa-gpu.log')
#logging.basicConfig(level=logging.INFO)
tsk = autotvm.task.Task(conv, [226, 226, 3, 3, 64, 64, 1, 'float32'], {})
tgt = tvm.target.create('cuda -model=p4000')
tsk.init_space(tgt, None)
print(tsk.config_space)

#tuner = autotvm.tuner.RandomTuner(tsk)
#tuner = autotvm.tuner.GATuner(tsk, pop_size=128, elite_num=4)
#tuner = autotvm.tuner.XGBTuner(tsk, tgt, 16, 4)
tuner = SATuner(tsk, 16, 960)

tuner.add_callback(autotvm.callback.SingleBestRecorder())

fmeature = autotvm.fleet.get_measure_batch(
    autotvm.fleet.create('titanx', timeout=20.0),
    tgt, 'llvm', repeat=5,
    retry_failures=None,
    replay_db=None,
    save=False,
    check_correctness=False)

logging.log(logging.INFO, str(len(tsk.config_space)))
start_time = time.time()
tuner.tune(fmeature, tgt, 800, 1)
logging.log(logging.INFO, str(time.time() - start_time))
