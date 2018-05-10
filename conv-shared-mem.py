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

def schedule_conv1_16(triple):
    neuron_i, synapse, neuron_n = triple

    sch = tvm.create_schedule(neuron_n.op)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 16), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")
    #thread_v = tvm.thread_axis((0, 9), "vthread", name = "vx")

    #shared_neuron = sch.cache_read(neuron_i, 'shared', [neuron_n])
    shared_synaps = sch.cache_read(synapse, 'shared', [neuron_n])
    #local_neuron  = sch.cache_read(shared_neuron, 'local', [neuron_n])
    local_synaps  = sch.cache_read(shared_synaps, 'local', [neuron_n])
    #local_output  = sch.cache_write(neuron_n, 'local')

    y, x, n, b = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    yo, yi = sch[neuron_n].split(y, factor = 8)
    xo, xi = sch[neuron_n].split(x, factor = 8)
    no, ni = sch[neuron_n].split(n, nparts = 16)
    #no, ni = sch[neuron_n].split(n, nparts = 32)
    io, ii = sch[neuron_n].split(i, 4)
    sch[neuron_n].reorder(yo, xo, no, ni, b, yi, xi, io, ky, kx, ii)

    #sch[shared_neuron].compute_at(sch[neuron_n], io)
    #ax0, ax1, ax2, ax3 = sch[shared_neuron].op.axis
    #ax3o, ax3i = sch[shared_neuron].split(ax3, 4)
    #sch[shared_neuron].vectorize(ax3i)

    sch[shared_synaps].compute_at(sch[neuron_n], b)
    ax0, ax1, ax2, ax3 = sch[shared_synaps].op.axis
    ax3o, ax3i = sch[shared_synaps].split(ax3, 4)
    sch[shared_synaps].bind(ax3o, block_x)
    sch[shared_synaps].vectorize(ax3i)
    #sch[shared_synaps].bind(ax1, thread_y)

    sch[local_synaps].compute_at(sch[neuron_n], kx)
    ax0, ax1, ax2, ax3 = sch[local_synaps].op.axis
    ax3o, ax3i = sch[local_synaps].split(ax3, 4)
    sch[local_synaps].vectorize(ax3i)

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

#41.81 ms
#cpu_args, gpu_args = prepare_args(224, 224, 16, 64, 64, 3, 3)
#conv1_16  = conv_layer(224, 224, 16, 64, 64, 3, 3)
#test_gpu(schedule_conv1_16(conv1_16), gpu_args)

def schedule_conv2_16(triple):
    neuron_i, synapse, neuron_n = triple

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis((0, 2), "blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch = tvm.create_schedule(neuron_n.op)

    #shared_neuron = sch.cache_read(neuron_i, 'shared', [neuron_n])
    #local_neuron  = sch.cache_read(shared_neuron, 'local', [neuron_n])
    shared_synaps = sch.cache_read(synapse, 'shared', [neuron_n])
    #local_synaps  = sch.cache_read(shared_synaps, 'local', [neuron_n])
    #local_output  = sch.cache_write(neuron_n, 'local')

    b, y, x, n = sch[neuron_n].op.axis
    #print(sch[neuron_n].op.axis)
    ky, kx, i = sch[neuron_n].op.reduce_axis
    yx = sch[neuron_n].fuse(y, x)
    yxo, yxi = sch[neuron_n].split(yx, nparts = 4)
    no, ni = sch[neuron_n].split(n, nparts = 128)
    io, ii = sch[neuron_n].split(i, 4)
    ioo, ioi = sch[neuron_n].split(io, 4)
    sch[neuron_n].reorder(yxo, yxi, no, b, ni, ky, kx, ioo, ioi, ii)
    #sch[neuron_n].reorder(yxo, yxi, no, b, ni, ky, kx, io, ii)

    sch[shared_synaps].compute_at(sch[neuron_n], kx)
    ax0, ax1, ax2, ax3 = shared_synaps.op.axis
    ax3o, ax3i = sch[shared_synaps].split(ax3, 4)
    sch[shared_synaps].vectorize(ax3i)
    sch[shared_synaps].bind(ax3o, block_x)
    #sch[shared_synaps].bind(ax2, block_z)


    sch[neuron_n].vectorize(ii)

    sch[neuron_n].bind(yxo, block_z)
    sch[neuron_n].bind(yxi, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(b, thread_y)
    sch[neuron_n].bind(ni, thread_x)

    print(tvm.lower(sch, triple, simple_mode = True))
    func = tvm.build(sch, triple, target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

#Latency: 14.52 ms
cpu_args, gpu_args = prepare_args(14, 14, 16, 512, 512, 3, 3, False)
conv2_16 = conv_layer(14, 14, 16, 512, 512, 3, 3, False)
test_gpu(schedule_conv2_16(conv2_16), gpu_args)

