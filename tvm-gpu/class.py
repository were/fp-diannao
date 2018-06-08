#!/usr/bin/env python3
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

    ro, ri = sch[temp].split(temp.op.reduce_axis[0], 4)
    roo, roi = sch[temp].split(ro, 4)
    sch[temp].vectorize(ri)
    sch[temp].unroll(roi)

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

#cpu_args, gpu_args = prepare_args(25088, 4096, 1)
#class1_1 = class_layer(25088, 4096, 1)
#Latency: 109.26 ms
#test_cpu(schedule_cpu(class1_1), cpu_args)
#Latency: 4.13 ms
#test_gpu(schedule_gpu1_1(class1_1), gpu_args)

def schedule_gpu1_16(four):
    neuron_i, synapse, temp, neuron_n = four
    sch = tvm.create_schedule(neuron_n.op)
    
    b, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    no, ni = sch[neuron_n].split(n, 49)
    noo, noi = sch[neuron_n].split(no, nparts = 16)
    nio, nii = sch[neuron_n].split(ni, nparts = 7)
    sch[temp].compute_at(sch[neuron_n], nii)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch[neuron_n].bind(noo, block_z)
    sch[neuron_n].bind(noi, block_y)
    sch[neuron_n].bind(b, block_x)
    sch[neuron_n].bind(nio, thread_y)
    sch[neuron_n].bind(nii, thread_x)

    ro, ri = sch[temp].split(temp.op.reduce_axis[0], 4)
    roo, roi = sch[temp].split(ro, 4)
    sch[temp].vectorize(ri)
    sch[temp].unroll(roi)
    #sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, four, simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU compilation done...')

    return func

#cpu_args, gpu_args = prepare_args(25088, 4096, 16)
#class1_16 = class_layer(25088, 4096, 16)
#Latency: 29.79 ms
#test_gpu(schedule_gpu1_16(class1_16), gpu_args)

def schedule_gpu1_32(four):
    neuron_i, synapse, temp, neuron_n = four
    sch = tvm.create_schedule(neuron_n.op)
    
    b, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    no, ni = sch[neuron_n].split(n, nparts = 49)
    noo, noi = sch[neuron_n].split(no, nparts = 7)
    nio, nii = sch[neuron_n].split(ni, nparts = 16)
    sch[temp].compute_at(sch[neuron_n], nii)
    bo, bi = sch[neuron_n].split(b, nparts = 16)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    sch[neuron_n].bind(noo, block_z)
    sch[neuron_n].bind(noi, block_y)
    sch[neuron_n].bind(bo, block_x)
    sch[neuron_n].bind(bi, thread_z)
    sch[neuron_n].bind(nio, thread_y)
    sch[neuron_n].bind(nii, thread_x)

    ro, ri = sch[temp].split(temp.op.reduce_axis[0], 4)
    roo, roi = sch[temp].split(ro, 4)
    sch[temp].vectorize(ri)
    sch[temp].unroll(roi)
    #sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, four, simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU compilation done...')

    return func

#cpu_args, gpu_args = prepare_args(25088, 4096, 32)
#class1_32 = class_layer(25088, 4096, 32)
#Latency: 96.48 ms
#test_gpu(schedule_gpu1_32(class1_32), gpu_args)

def schedule_gpu2_1(four):
    neuron_i, synapse, temp, neuron_n = four
    sch = tvm.create_schedule(neuron_n.op)
    
    b, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    no, ni = sch[neuron_n].split(n, nparts = 64)
    noo, noi = sch[neuron_n].split(no, nparts = 8)
    nio, nii = sch[neuron_n].split(ni, nparts = 8)
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

#cpu_args, gpu_args = prepare_args(4096, 1024, 1)
#class2_1 = class_layer(4096, 1024, 1)
#Latency: 4.37ms
#test_cpu(schedule_cpu(class2_1), cpu_args)
#Latency: 0.35ms
#test_gpu(schedule_gpu2_1(class2_1), gpu_args)

def schedule_gpu2_2(four):
    neuron_i, synapse, temp, neuron_n = four
    sch = tvm.create_schedule(neuron_n.op)
    
    b, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    no, ni = sch[neuron_n].split(n, nparts = 64)
    noo, noi = sch[neuron_n].split(no, nparts = 8)
    nio, nii = sch[neuron_n].split(ni, nparts = 8)
    sch[temp].compute_at(sch[neuron_n], nii)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    ro, ri = sch[temp].split(temp.op.reduce_axis[0], 4)
    roo, roi = sch[temp].split(ro, 4)
    sch[temp].vectorize(ri)
    sch[temp].unroll(roi)

    bnoo = sch[neuron_n].fuse(b, noo)
    sch[neuron_n].bind(bnoo, block_y)
    sch[neuron_n].bind(noi, block_x)
    sch[neuron_n].bind(nio, thread_y)
    sch[neuron_n].bind(nii, thread_x)

    #sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, four, simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU compilation done...')

    return func

#cpu_args, gpu_args = prepare_args(4096, 1024, 2)
#class2_2 = class_layer(4096, 1024, 2)
#Latency: 0.18ms
#test_gpu(schedule_gpu2_2(class2_2), gpu_args)

def schedule_gpu2_4(four):
    neuron_i, synapse, temp, neuron_n = four
    sch = tvm.create_schedule(neuron_n.op)
    
    b, n = sch[neuron_n].op.axis
    print(sch[neuron_n].op.axis)
    bo, bi = sch[neuron_n].split(b, nparts = 16)
    no, ni = sch[neuron_n].split(n, nparts = 64)
    #noo, noi = sch[neuron_n].split(no, nparts = 8)
    #nio, nii = sch[neuron_n].split(ni, nparts = 8)
    sch[temp].compute_at(sch[neuron_n], ni)

    ro, ri = sch[temp].split(temp.op.reduce_axis[0], 4)
    sch[temp].vectorize(ri)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    #shared_neuron = sch.cache_read(neuron_i, 'shared', [temp])
    #sch[shared_neuron].compute_at(sch[neuron_n], nii)
    #ax0, ax1 = shared_neuron.op.axis
    #ax0o, ax0i = sch[shared_neuron].split(ax0, 8)
    #ax0io, ax0ii = sch[shared_neuron].split(ax0i, 4)
    #sch[shared_neuron].unroll(ax1)
    #sch[shared_neuron].unroll(ax0io)
    #sch[shared_neuron].vectorize(ax0ii)
    #ax0oo, ax0oi = sch[shared_neuron].split(ax0o, 32)
    #sch[shared_neuron].bind(ax0oi, thread_x)
    #sch[shared_neuron].bind(ax0oo, block_x)

    #local_neuron = sch.cache_read(shared_neuron, 'local', [temp])
    #sch[local_neuron].compute_at(sch[temp], ro)
    #ax0, ax1 = local_neuron.op.axis
    #sch[local_neuron].vectorize(ax0)

    #sch[neuron_n].bind(noo, block_y)
    #sch[neuron_n].bind(noi, block_x)
    #sch[neuron_n].bind(nio, thread_y)
    #sch[neuron_n].bind(nii, thread_x)

    sch[neuron_n].bind(no, block_y)
    sch[neuron_n].bind(ni, block_z)
    sch[neuron_n].bind(bo, thread_y)
    sch[neuron_n].bind(bi, thread_x)

    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, 8), "threadIdx.x")
    thread_y = tvm.thread_axis((0, 8), "threadIdx.y")
    thread_z = tvm.thread_axis((0, 8), "threadIdx.z")

    #sch[neuron_n].reorder(y, x, ky, kx, i, n, b)
    
    print(tvm.lower(sch, four, simple_mode = True))
    func = tvm.build(sch, [neuron_i, synapse, neuron_n], target = 'cuda')
    assert func
    print('GPU compilation done...')

    return func

cpu_args, gpu_args = prepare_args(4096, 1024, 512)
class2_4 = class_layer(4096, 1024, 512)
##Latency: 4.8 ms
test_gpu(schedule_gpu2_4(class2_4), gpu_args)
