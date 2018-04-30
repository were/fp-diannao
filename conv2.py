import numpy
import tvm

dtype = 'float32'

batch = 1

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
    sch[neuron_n].reorder(b, y, x, ky, kx, i, n)
    
    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'llvm')
    assert func
    print('CPU compilation done...')

    nd_neuron_i = tvm.nd.array(np_neuron_i, tvm.cpu(0))
    nd_synapse  = tvm.nd.array(np_synapse, tvm.cpu(0))
    nd_neuron_n = tvm.nd.array(np_neuron_n, tvm.cpu(0))
    
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
    print('CPU Convolution: %.2f ms' % (evaluator(nd_neuron_i, nd_synapse, nd_neuron_n).mean * 1000))

def test_gpu():
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
    sch[neuron_n].reorder(b, yx, no, ni, ky, kx, i)
    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    sch[neuron_n].bind(yx, block_y)
    sch[neuron_n].bind(no, block_x)
    sch[neuron_n].bind(ni, thread_x)

    print(tvm.lower(sch, [neuron_i, synapse, neuron_n], simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU Compilation Done...')
    
    nd_neuron_i = tvm.nd.array(np_neuron_i, tvm.gpu(0))
    nd_synapse  = tvm.nd.array(np_synapse, tvm.gpu(0))
    nd_neuron_n = tvm.nd.array(np_neuron_n, tvm.gpu(0))
    
    evaluator = func.time_evaluator(func.entry_name, tvm.gpu(0), number = 5)
    print('GPU Convolution: %.2f ms' % (evaluator(nd_neuron_i, nd_synapse, nd_neuron_n).mean * 1000))

#test_cpu()
test_gpu()


