import torch

def adjust_lr(init_lr, optimizer, epoch, half_rate=100):
    #TODO: considering using lambda func to obtain a more general way to adjust learning rate
    """adjust learning rate for the parameters on the optimizer

    Args:
        cur_lr (float): current learning rate
        optimizer (torch.optimizer): change the learning rate of this optimizer
        epoch (int): current epoch
        half_rate (optional:int default=100): when epoch
    """
    lr = init_lr * (0.5 ** (epoch // half_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(net,save_path):
    torch.save(net.state_dict(),save_path)

