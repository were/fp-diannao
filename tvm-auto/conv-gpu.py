#!/usr/bin/env python2

import topi, tvm, autotvm, time, numpy, logging, operator
from autotvm import MeasureResult, MeasureErrorNo
from sa import SATuner
from ga import GATuner

@autotvm.simple_template
def conv(iw, ih, fw, fh, fi, fo, batch, dtype):
    img = tvm.placeholder((batch, fi, iw, ih), dtype=dtype, name='img')
    fil = tvm.placeholder((fi, fo, fw, fh), dtype=dtype, name='fil')

    conv= topi.nn.conv2d_nchw(img, fil, (1, 1), 'VALID')

    cfg = autotvm.template.DispatchContext.current.query(None, None)

    cfg.add_flop(iw * ih * fw * fh * fi * fo * batch * 2)

    s = tvm.create_schedule(conv.op)

    temp = conv.op.input_tensors[0]
    sch[temp].compute_inline()

    shared_cache = []
    local_cache  = []

    # Space definition
    for buf in conv.op.input_tensors:
        shared_cache.append(s.cache_read(buf, "shared", [conv]))
        local_cache.append(s.cache_read(shared_cache[-1], "local", [conv]))
    write_cache = s.cache_write(conv, "local")

    spatial_axes = [cfg.axis(x) for x in s[conv].op.axis]
    spatial_chs = [cfg.define_split("tile_" + x.name, x, num_outputs=4)
                   for x in spatial_axes]
    re_axes = cfg.define_reorder("re",
                          reduce(list.__add__, spatial_chs),
                          policy='interleave', spatial=spatial_chs, reduce=[])
    cfg.define_annotate('bind', re_axes[:sum([len(ch) - 1 for ch in spatial_chs])],
                 policy='bind_gpu_virtual')

    reduce_axes = [cfg.axis(x) for x in s[write_cache].op.reduce_axis]
    reduce_chs = [cfg.define_split("tile_reduce_" + x.name, x, num_outputs=2)
                  for x in reduce_axes]
    cfg.define_annotate("cache_anchor", reduce(list.__add__, reduce_chs),
                 policy='locate_cache', num_anchor=2)

    # Apply on schedule
    spatial_axes = s[conv].op.axis
    spatial_chs = [cfg["tile_" + x.var.name].apply(s, conv, x)
                   for x in spatial_axes]
    spatial_lens = [cfg["tile_" + x.var.name].size
                    for x in spatial_axes]

    re_axes = cfg["re"].apply(s, conv, reduce(list.__add__, spatial_chs))
    bind_axes = re_axes[:sum([len(ch) - 1 for ch in spatial_chs])]
    cfg['bind'].apply(s, conv, bind_axes)

    # Cache anchor
    s[write_cache].compute_at(s[conv], bind_axes[-1])

    local_axes = s[write_cache].op.axis
    reduce_axes = s[write_cache].op.reduce_axis
    reduce_chs = [cfg["tile_reduce_" + x.var.name].apply(s, write_cache, x)
                  for x in reduce_axes]
    s[write_cache].reorder(*(reduce(list.__add__, reduce_chs) + list(local_axes)))
    cfg['cache_anchor'].apply(s, write_cache, reduce(list.__add__, reduce_chs),
                                       source=[shared_cache, local_cache])

    re_lens = [reduce(list.__add__, spatial_lens)[x] for x in cfg["re"].perm]
    bind_lens = re_lens[:sum([len(ch) - 1 for ch in spatial_chs])]
    thread_info = []
    for ann, length in zip(cfg['bind'].anns, bind_lens):
        if 'threadIdx' in ann:
            thread_info.append((ann, length))
    thread_info.sort(key=lambda x: x[0])

    for i, cache in enumerate(shared_cache):
        axes = list(s[cache].op.axis)
        fused = s[cache].fuse(*axes)
        for name, length in reversed(thread_info):
            t, fused = s[cache].split(fused, nparts=length)
            s[cache].bind(t, tvm.thread_axis(name))

    return s, [img, fil, conv]

logging.basicConfig(level=logging.INFO, filename='ga-conv1.log')
#logging.basicConfig(level=logging.INFO)
#tsk = autotvm.task.Task(conv, [226, 226, 3, 3, 64, 64, 1, 'float32'], {})
tsk = autotvm.task.create_task('conv', [226, 226, 3, 3, 64, 64, 1, 'float32'], 'cuda', 'llvm')
#tsk = autotvm.task.create_task('conv', [17, 17, 3, 3, 512, 512, 1, 'float32'], 'cuda', 'llvm')
tgt = tvm.target.create('cuda -model=p4000')
tsk.init_space(tgt, None)
print(tsk.config_space)

tuner = GATuner(tsk, n_pool=32, n_elites=4)
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
