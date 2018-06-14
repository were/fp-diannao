#!/usr/bin/env python3
import numpy, tvm, argparse

dtype = 'float32'

def conv_layer(Nx, Ny, batch, Ni, Nn, Kx, Ky, batch_inner = True):
    NyPad = Ny + Ky - 1
    NxPad = Nx + Kx - 1

    neuron_i = tvm.placeholder((NyPad, NxPad, Ni, batch), name = 'neuron_i', dtype = dtype)
    synapse  = tvm.placeholder((Ny, Nx, Ni, Nn), name = 'synapse', dtype = dtype)

    ky = tvm.reduce_axis((0, Ky), name='ky')
    kx = tvm.reduce_axis((0, Kx), name='kx')
    i = tvm.reduce_axis((0, Ni), name='i')

    if batch_inner:
        neuron_n = tvm.compute(
            (Ny, Nx, Nn, batch),
            lambda y, x, n, b:
                tvm.sum(synapse[ky][kx][i][n] * neuron_i[y + ky][x + kx][i][b],
            axis=[ky, kx, i]),
            name='neuron_n'
        )
    else:
        neuron_n = tvm.compute(
            (batch, Ny, Nx, Nn),
            lambda b, y, x, n:
                tvm.sum(synapse[ky][kx][i][n] * neuron_i[y + ky][x + kx][i][b],
            axis=[ky, kx, i]),
            name='neuron_n'
        )

    return neuron_i, synapse, neuron_n


def prepare_args(Nx, Ny, batch, Ni, Nn, Kx, Ky, batch_inner = True):
    NyPad = Ny + Ky - 1
    NxPad = Nx + Kx - 1

    np_neuron_i = numpy.random.uniform(size = (NyPad, NxPad, Ni, batch)).astype(dtype)
    np_synapse  = numpy.random.uniform(size = (Ny, Nx, Ni, Nn)).astype(dtype)

    out_shape = (Ny, Nx, Nn, batch) if batch_inner else (batch, Ny, Nx, Nn)
    np_neuron_n = numpy.zeros(out_shape).astype(dtype)

    cpu_args = [
        tvm.nd.array(np_neuron_i, tvm.cpu(0)),
        tvm.nd.array(np_synapse, tvm.cpu(0)),
        tvm.nd.array(np_neuron_n, tvm.cpu(0))
    ]

    print("args prepared")
    
    return cpu_args


def schedule_cpu(triple):
    neuron_i, synapse, neuron_n = triple
    sch = tvm.create_schedule(neuron_n.op)
    
    y, x, n, b = sch[neuron_n].op.axis
    ky, kx, i = sch[neuron_n].op.reduce_axis
    no, ni = sch[neuron_n].split(n, 8)
    #io, ii = sch[neuron_n].split(i, 8)
    yo, yi = sch[neuron_n].split(y, 14)
    xo, xi = sch[neuron_n].split(x, 8)
    sch[neuron_n].vectorize(ni)
    #sch[neuron_n].reorder(yo, xo, yi, ky, kx, xi, io, ii, no, ni)
    sch[neuron_n].reorder(yo, xo, yi, ky, kx, xi, i, no, ni)
    #yxo = sch[neuron_n].fuse(yo, xo)
    #sch[neuron_n].parallel(yo)


    print(tvm.lower(sch, triple, simple_mode = True))
    func = tvm.build(sch, triple, target = 'llvm')
    assert func
    print('CPU compilation done...')

    return func

def test_cpu(func, total, cpu_args):
    evaluator = func.time_evaluator(func.entry_name, tvm.cpu(0), number = 5)
    tic = evaluator(*cpu_args).mean
    print('CPU Convolution: %f GFLOP/s' % (total / tic / 1e9))
    print('CPU Convolution: %f ms' % (tic * 1000.))


conv_flops = 224 * 224 * 1 * 64 * 64 * 3 * 3
#Latency: 987.99ms
cpu_args = prepare_args(224, 224, 1, 64, 64, 3, 3)
conv1    = conv_layer(224, 224, 1, 64, 64, 3, 3)
#13.3 GFLOP/s
test_cpu(schedule_cpu(conv1), conv_flops, cpu_args)

#Latency: 8.77 ms
#cpu_args, gpu_args = prepare_args(14, 14, 1, 512, 512, 3, 3)
#conv2_1  = conv_layer(14, 14, 1, 512, 512, 3, 3)
#test_gpu(schedule_conv2_1(conv2_1), gpu_args)

