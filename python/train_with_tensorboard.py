import sys
sys.path.append('/home/tumh/caffe-tensorboard/python')
sys.path.append('/home/tumh/tensorboard-pytorch')
import caffe
from tensorboardX import SummaryWriter

solver = caffe.SGDSolverWrapper('/home/tumh/caffe-tensorboard/examples/mnist/lenet_solver.prototxt')

writer = SummaryWriter()

n_iter = 0
while True:
    loss = 0
    for i in range(solver.param.iter_size):
        loss += solver.forward_backward()
    if n_iter % 100 == 0:
        writer.add_scalar('data/loss', loss, n_iter)
    solver.update_smoothloss(loss, solver.param.average_loss)
    solver.apply_update()
    n_iter = n_iter + 1

writer.close()
