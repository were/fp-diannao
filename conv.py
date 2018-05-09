import numpy, tvm, argparse

dtype = 'float32'

def conv_layer(Nx, Ny, batch, Ni, Nn, Kx, Ky, batch_inner = True):
    NyPad = Ny + Ky - 1
    NxPad = Nx + Kx - 1

    neuron_i = tvm.placeholder((NyPad, NxPad, Ni, batch), name = 'neuron_i', dtype = dtype)
    synapse  = tvm.placeholder((Ny, Nx, Nn, Ni), name = 'synapse', dtype = dtype)

    ky = tvm.reduce_axis((0, Ky), name='ky')
    kx = tvm.reduce_axis((0, Kx), name='kx')
    i = tvm.reduce_axis((0, Ni), name='i')

    if batch_inner:
        neuron_n = tvm.compute(
            (Ny, Nx, Nn, batch),
            lambda y, x, n, b:
                tvm.sum(synapse[ky][kx][n][i] * neuron_i[y + ky][x + kx][i][b],
            axis=[ky, kx, i]),
            name='neuron_n'
        )
    else:
        neuron_n = tvm.compute(
            (batch, Ny, Nx, Nn),
            lambda b, y, x, n:
                tvm.sum(synapse[ky][kx][n][i] * neuron_i[y + ky][x + kx][i][b],
            axis=[ky, kx, i]),
            name='neuron_n'
        )

    print(tvm.lower(tvm.create_schedule(neuron_n.op), [neuron_i, synapse, neuron_n], simple_mode = True))
    return neuron_i, synapse, neuron_n


def prepare_args(Nx, Ny, batch, Ni, Nn, Kx, Ky, batch_inner = True):
    NyPad = Ny + Ky - 1
    NxPad = Nx + Kx - 1

    np_neuron_i = numpy.random.uniform(size = (NyPad, NxPad, Ni, batch)).astype(dtype)
    np_synapse  = numpy.random.uniform(size = (Ny, Nx, Nn, Ni)).astype(dtype)

    out_shape = (Ny, Nx, Nn, batch) if batch_inner else (batch, Ny, Nx, Nn)
    np_neuron_n = numpy.zeros(out_shape).astype(dtype)

    cpu_args = [
        tvm.nd.array(np_neuron_i, tvm.cpu(0)),
        tvm.nd.array(np_synapse, tvm.cpu(0)),
        tvm.nd.array(np_neuron_n, tvm.cpu(0))
    ]
    
    gpu_args = [
        tvm.nd.array(np_neuron_i, tvm.gpu(0)),
        tvm.nd.array(np_synapse, tvm.gpu(0)),
        tvm.nd.array(np_neuron_n, tvm.gpu(0))
    ]

    return cpu_args, gpu_args


def schedule_cpu(triple):
    neuron_i, synapse, neuron_n = triple
    sch = tvm.create_schedule(neuron_n.op)
    
    y, x, n, b = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, triple, simple_mode = True))
    func = tvm.build(sch, triple, target = 'llvm')
    assert func
    print('CPU compilation done...')

    return func

def test_cpu(func, cpu_args):
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
    ms = evaluator(*cpu_args).mean
    print('CPU Convolution: %.2f ms' % (ms * 1000))


def test_gpu(func, gpu_args):
    #print(func.imported_modules[0].get_source())
    evaluator = func.time_evaluator(func.entry_name, tvm.gpu(0), number = 5)
    print('GPU Convolution: %.2f ms' % (evaluator(*gpu_args).mean * 1000))

def schedule_conv1_1(triple):
    neuron_i, synapse, neuron_n = triple

    sch = tvm.create_schedule(neuron_n.op)

    #TBD: shared memory
    #shared_neuron = sch.cache_read(neuron_i, 'shared', [neuron_n])
    #shared_synaps = sch.cache_read(synapse, 'shared', [neuron_n])
    #local_neuron  = sch.cache_read(shared_neuron, 'local', [neuron_n])
    #local_synaps  = sch.cache_read(shared_synaps, 'local', [neuron_n])
    #local_output  = sch.cache_write(neuron_n, 'local')

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    y, x, n, b = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    yo, yi = sch[neuron_n].split(y, nparts = 16)
    xo, xi = sch[neuron_n].split(x, nparts = 16)
    no, ni = sch[neuron_n].split(n, nparts = 16)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(yo, xo, no, yi, xi, ni, ky, kx, io, ii)

    #sch[shared_synaps].compute_at(sch[neuron_n], no)
    #ax0, ax1, ax2, ax3 = sch[shared_synaps].op.axis
    #ax3o, ax3i = sch[shared_synaps].split(ax3, 4)
    #sch[shared_synaps].vectorize(ax3i)

    #sch[local_synaps].compute_at(sch[neuron_n], io)
    #ax0, ax1, ax2, ax3 = sch[local_synaps].op.axis
    #sch[local_synaps].vectorize(ax3)

    sch[neuron_n].vectorize(ii)

    sch[neuron_n].bind(yo, block_z)
    sch[neuron_n].bind(xo, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(yi, thread_z)
    sch[neuron_n].bind(xi, thread_y)
    sch[neuron_n].bind(ni, thread_x)

    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#cpu_args, gpu_args = prepare_args(224, 224, 1, 64, 64, 3, 3)
#Latency: 987.99ms
#test_cpu(schedule_cpu(conv1_1), cpu_args)
#Latency: 10.02 ms
#conv1_1  = conv_layer(224, 224, 1, 64, 64, 3, 3)
#test_gpu(schedule_conv1_1(conv1_1), gpu_args)

def schedule_conv1_2(triple):
    neuron_i, synapse, neuron_n = triple

    sch = tvm.create_schedule(neuron_n.op)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    b, y, x, n = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    yo, yi = sch[neuron_n].split(y, nparts = 32)
    xo, xi = sch[neuron_n].split(x, nparts = 32)
    no, ni = sch[neuron_n].split(n, nparts = 16)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(yo, xo, no, yi, xi, ni, b, ky, kx, io, ii)
    sch[neuron_n].vectorize(ii)
    nib = sch[neuron_n].fuse(ni, b)
    sch[neuron_n].bind(yo, block_z)
    sch[neuron_n].bind(xo, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(yi, thread_z)
    sch[neuron_n].bind(xi, thread_y)
    sch[neuron_n].bind(nib, thread_x)

    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#13.34ms
#cpu_args, gpu_args = prepare_args(224, 224, 2, 64, 64, 3, 3, False)
#conv1_2  = conv_layer(224, 224, 2, 64, 64, 3, 3, False)
#test_gpu(schedule_conv1_2(conv1_2), gpu_args)

def schedule_conv1_8(triple):
    neuron_i, synapse, neuron_n = triple

    sch = tvm.create_schedule(neuron_n.op)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 16), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    b, y, x, n = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    yo, yi = sch[neuron_n].split(y, nparts = 32)
    xo, xi = sch[neuron_n].split(x, nparts = 32)
    no, ni = sch[neuron_n].split(n, nparts = 32)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(yo, xo, no, yi, xi, b, ni, ky, kx, io, ii)
    sch[neuron_n].vectorize(ii)
    sch[neuron_n].bind(yo, block_z)
    sch[neuron_n].bind(xo, block_y)
    sch[neuron_n].bind(no, block_x)
    bni = sch[neuron_n].fuse(b, ni)
    sch[neuron_n].bind(yi, thread_z)
    sch[neuron_n].bind(xi, thread_y)
    sch[neuron_n].bind(bni, thread_x)

    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#37.23 ms
#cpu_args, gpu_args = prepare_args(224, 224, 8, 64, 64, 3, 3, False)
#conv1_8  = conv_layer(224, 224, 8, 64, 64, 3, 3, False)
#test_gpu(schedule_conv1_8(conv1_8), gpu_args)

def schedule_conv1_16(triple):
    neuron_i, synapse, neuron_n = triple

    sch = tvm.create_schedule(neuron_n.op)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 16), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    b, y, x, n = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    yo, yi = sch[neuron_n].split(y, nparts = 32)
    xo, xi = sch[neuron_n].split(x, nparts = 32)
    no, ni = sch[neuron_n].split(n, nparts = 32)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(yo, xo, no, yi, xi, b, ni, ky, kx, io, ii)
    sch[neuron_n].vectorize(ii)
    sch[neuron_n].bind(yo, block_z)
    sch[neuron_n].bind(xo, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(yi, thread_z)
    sch[neuron_n].bind(xi, thread_y)
    sch[neuron_n].bind(b, thread_x)

    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#58.87 ms
#cpu_args, gpu_args = prepare_args(224, 224, 16, 64, 64, 3, 3, False)
#conv1_16  = conv_layer(224, 224, 16, 64, 64, 3, 3, False)
#test_gpu(schedule_conv1_16(conv1_16), gpu_args)

def schedule_conv2_1(triple):
    neuron_i, synapse, neuron_n = triple

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch = tvm.create_schedule(neuron_n.op)

    print(sch[neuron_n].op.axis)
    y, x, n, b = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    no, ni = sch[neuron_n].split(n, nparts = 32)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(y, x, no, ni, ky, kx, io, ii, b)
    sch[neuron_n].vectorize(ii)
    sch[neuron_n].bind(y, block_y)
    sch[neuron_n].bind(x, block_x)
    sch[neuron_n].bind(no, thread_y)
    sch[neuron_n].bind(ni, thread_x)

    print(tvm.lower(sch, triple, simple_mode = True))
    func = tvm.build(sch, triple, target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#Latency: 8.77 ms
#cpu_args, gpu_args = prepare_args(14, 14, 1, 512, 512, 3, 3)
#conv2_1  = conv_layer(14, 14, 1, 512, 512, 3, 3)
#test_gpu(schedule_conv2_1(conv2_1), gpu_args)


def schedule_conv2_2(triple):
    neuron_i, synapse, neuron_n = triple

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch = tvm.create_schedule(neuron_n.op)

    b, y, x, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    ky, kx, i = sch[neuron_n].op.reduce_axis
    #yx = sch[neuron_n].fuse(y, x)
    no, ni = sch[neuron_n].split(n, nparts = 32)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(y, x, no, ni, b, ky, kx, io, ii)
    sch[neuron_n].vectorize(ii)
    nib = sch[neuron_n].fuse(ni, b)
    sch[neuron_n].bind(y, block_y)
    sch[neuron_n].bind(x, block_x)
    sch[neuron_n].bind(no, thread_y)
    sch[neuron_n].bind(nib, thread_x)

    print(tvm.lower(sch, triple, simple_mode = True))
    func = tvm.build(sch, triple, target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#Latency: 10.32 ms
#cpu_args, gpu_args = prepare_args(14, 14, 2, 512, 512, 3, 3, False)
#conv2_2  = conv_layer(14, 14, 2, 512, 512, 3, 3, False)
#test_gpu(schedule_conv2_2(conv2_2), gpu_args)

def schedule_conv2_8(triple):
    neuron_i, synapse, neuron_n = triple

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch = tvm.create_schedule(neuron_n.op)

    b, y, x, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    ky, kx, i = sch[neuron_n].op.reduce_axis
    yx = sch[neuron_n].fuse(y, x)
    no, ni = sch[neuron_n].split(n, nparts = 64)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(yx, no, b, ni, ky, kx, io, ii)
    sch[neuron_n].vectorize(ii)
    sch[neuron_n].bind(yx, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(b, thread_y)
    sch[neuron_n].bind(ni, thread_x)

    print(tvm.lower(sch, triple, simple_mode = True))
    func = tvm.build(sch, triple, target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#Latency: 27.96 ms
#cpu_args, gpu_args = prepare_args(14, 14, 8, 512, 512, 3, 3, False)
#conv2_8  = conv_layer(14, 14, 8, 512, 512, 3, 3, False)
#test_gpu(schedule_conv2_8(conv2_8), gpu_args)

def schedule_conv2_16(triple):
    neuron_i, synapse, neuron_n = triple

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch = tvm.create_schedule(neuron_n.op)

    b, y, x, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    ky, kx, i = sch[neuron_n].op.reduce_axis
    #yx = sch[neuron_n].fuse(y, x)
    no, ni = sch[neuron_n].split(n, nparts = 128)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(y, x, no, b, ni, ky, kx, io, ii)
    sch[neuron_n].vectorize(ii)
    sch[neuron_n].bind(y, block_z)
    sch[neuron_n].bind(x, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(b, thread_y)
    sch[neuron_n].bind(ni, thread_x)

    print(tvm.lower(sch, triple, simple_mode = True))
    func = tvm.build(sch, triple, target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#Latency: 34.43 ms
cpu_args, gpu_args = prepare_args(14, 14, 16, 512, 512, 3, 3, False)
conv2_16 = conv_layer(14, 14, 16, 512, 512, 3, 3, False)
test_gpu(schedule_conv2_16(conv2_16), gpu_args)

