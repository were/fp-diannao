import topi, tvm


def dense(n, m, l):
    img = tvm.placeholder((n, m), dtype='float32')
    wei = tvm.placeholder((l, m), dtype='float32')
    res = topi.nn.dense(img, wei)

    sch = tvm.create_schedule(res.op)

    print(tvm.lower(sch, [img, wei, res], simple_mode=True))


dense(25088, 4096, 32)

