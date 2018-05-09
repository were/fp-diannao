import numpy, tvm, argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--config', action = 'store', type=int,
    dest = 'config', default = 1,
    help = 'The configuration of the convolution to be explored.'
)

parser.add_argument(
    '--batch', action = 'store', type=int,
    dest = 'batch', default = 1,
    help = 'The batch size of the computaion'
)

parser.add_argument(
    '--show-gpu-src', action = 'store_true',
    dest = 'show_gpu_src',
    help = 'The batch size of the computaion'
)

parser.add_argument(
    '--disable-gpu', action = 'store_true',
    dest = 'no_gpu',
    help = 'The batch size of the computaion'
)

parser.add_argument(
    '--disable-cpu', action = 'store_true',
    dest = 'no_cpu',
    help = 'The batch size of the computaion'
)


args = vars(parser.parse_args())
print(args)

dtype = 'float32'

batch = args.get('batch')

if args.get('config') == 1:
    Ny = 224
    Nx = 224
    Ni = 64
    Nn = 64

    Kx = 3
    Ky = 3
else:
    Ny = 14
    Nx = 14
    Ni = 512
    Nn = 512

    Kx = 3
    Ky = 3

NyPad = Ny + Ky - 1
NxPad = Nx + Kx - 1

neuron_i = tvm.placeholder((NyPad, NxPad, Ni, batch), name = 'neuron_i', dtype = dtype)
synapse  = tvm.placeholder((Ny, Nx, Nn, Ni), name = 'synapse', dtype = dtype)

ky = tvm.reduce_axis((0, Ky), name='ky')
kx = tvm.reduce_axis((0, Kx), name='kx')
i = tvm.reduce_axis((0, Ni), name='i')

neuron_n = tvm.compute(
        (Ny, Nx, Nn, batch), 
        lambda y, x, n, b:
            tvm.sum(synapse[ky][kx][n][i] * neuron_i[y + ky][x + kx][i][b],
        axis=[ky, kx, i]),
        name='neuron_n'
)

np_neuron_i = numpy.random.uniform(size = (NyPad, NxPad, Ni, batch)).astype(dtype)
np_synapse  = numpy.random.uniform(size = (Ny, Nx, Nn, Ni)).astype(dtype)
np_neuron_n = numpy.zeros((Ny, Nx, Nn, batch)).astype(dtype)

print('Data preparation done...')

def test_cpu():
    sch = tvm.create_schedule(neuron_n.op)
    
    y, x, n, b = sch[neuron_n].op.axis
    sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'llvm')
    assert func
    print('CPU compilation done...')

    nd_neuron_i = tvm.nd.array(np_neuron_i, tvm.cpu(0))
    nd_synapse  = tvm.nd.array(np_synapse, tvm.cpu(0))
    nd_neuron_n = tvm.nd.array(np_neuron_n, tvm.cpu(0))
    
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
    print('CPU Convolution: %.2f ms' % (evaluator(nd_neuron_i, nd_synapse, nd_neuron_n).mean * 1000))


def schedule_conv2():
    sch = tvm.create_schedule(neuron_n.op)
    
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    y, x, n, b = sch[neuron_n].op.axis
    yx = sch[neuron_n].fuse(y, x)
    no, ni = sch[neuron_n].split(n, nparts = 32)
    sch[neuron_n].reorder(yx, no, ni, ky, kx, i, b)
    sch[neuron_n].bind(yx, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(ni, thread_x)

    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

def schedule_conv1():
    sch = tvm.create_schedule(neuron_n.op)
    
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
    yo, yi = sch[neuron_n].split(y, nparts = 32)
    xo, xi = sch[neuron_n].split(x, nparts = 32)
    no, ni = sch[neuron_n].split(n, nparts = 8)
    sch[neuron_n].reorder(yo, xo, yi, xi, no, ni, ky, kx, i, b)
    sch[neuron_n].bind(yo, block_z)
    sch[neuron_n].bind(xo, block_y)
    sch[neuron_n].bind(yi, block_x)
    sch[neuron_n].bind(xi, thread_y)
    sch[neuron_n].bind(no, thread_x)

    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    return func

def test_gpu(func):
    if args['show_gpu_src']:
        print(func.imported_modules[0].get_source())

    nd_neuron_i = tvm.nd.array(np_neuron_i, tvm.gpu(0))
    nd_synapse  = tvm.nd.array(np_synapse, tvm.gpu(0))
    nd_neuron_n = tvm.nd.array(np_neuron_n, tvm.gpu(0))
    
    evaluator = func.time_evaluator(func.entry_name, tvm.gpu(0), number = 5)
    print('GPU Convolution: %.2f ms' % (evaluator(nd_neuron_i, nd_synapse, nd_neuron_n).mean * 1000))

if not args['no_cpu']:
    test_cpu()

if not args['no_gpu']:
    test_gpu(eval('schedule_conv%d()' % args.get('config')))

