import numpy, tvm, argparse

dtype = 'float32'

def class_layer(Nn, Ni, batch):
    synapse  = tvm.placeholder((Nn, Ni), name = 'synapse', dtype = dtype)
    neuron_i = tvm.placeholder((Ni, batch), name = 'neuron_i', dtype = dtype)

    i = tvm.reduce_axis((0, Ni), name='i')

    temp = tvm.compute(
        (batch, Nn),
        lambda b, n:
            tvm.sum(synapse[n][i] * neuron_i[i][b],
            axis=[i]),
        name='temp'
    )

    neuron_n = tvm.compute(
        (batch, Nn),
        lambda b, n:
            tvm.select(temp[b][n] > 0., temp[b][n], temp[b][n] / 4.0),
        name='neuron_n'
    )

    print(tvm.lower(tvm.create_schedule(neuron_n.op), [neuron_i, synapse, neuron_n], simple_mode = True))
    return neuron_i, synapse, temp, neuron_n

def prepare_args(Nn, Ni, batch):
    np_neuron_i = numpy.random.uniform(size = (Ni, batch)).astype(dtype)
    np_synapse  = numpy.random.uniform(size = (Nn, Ni)).astype(dtype)

    np_neuron_n = numpy.zeros((batch, Nn)).astype(dtype)

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


def schedule_cpu(four):
    neuron_i, synapse, temp, neuron_n = four
    sch = tvm.create_schedule(neuron_n.op)
    
    b, n = sch[neuron_n].op.axis
    sch[temp].compute_at(sch[neuron_n], n)
    #sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, four, simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'llvm')
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


def schedule_gpu1_1(four):
    neuron_i, synapse, temp, neuron_n = four
    sch = tvm.create_schedule(neuron_n.op)
    
    b, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    no, ni = sch[neuron_n].split(n, nparts = 49)
    noo, noi = sch[neuron_n].split(no, nparts = 7)
    nio, nii = sch[neuron_n].split(ni, nparts = 16)
    sch[temp].compute_at(sch[neuron_n], nii)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch[neuron_n].bind(noo, block_y)
    sch[neuron_n].bind(noi, block_x)
    sch[neuron_n].bind(nio, thread_y)
    sch[neuron_n].bind(nii, thread_x)

    #sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, four, simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU compilation done...')

    return func

cpu_args, gpu_args = prepare_args(25088, 4096, 1)
class1_1 = class_layer(25088, 4096, 1)
#Latency: 109.26 ms
#test_cpu(schedule_cpu(class1_1), cpu_args)
#Latency: 8.03 ms
test_gpu(schedule_gpu1_1(class1_1), gpu_args)

