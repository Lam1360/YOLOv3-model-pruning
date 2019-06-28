def inspect_lr(optimizer):
    cur_lr = optimizer.param_groups[0]['lr']
    print('Current learning rate: %0.6f' % cur_lr)
    return cur_lr

# 将学习率衰减为原来的gamma倍数
def modify_lr(optimizer, gamma):
    cur_lr = inspect_lr(optimizer)
    new_lr = cur_lr * gamma
    print('Learning rate has been changed from %0.6f to %0.6f' % (cur_lr, new_lr))
    for group in optimizer.param_groups:
        group['lr'] = new_lr
    return new_lr

def turn_on_sr(opt):
    opt.sr = True
    print('Sr has been turned on!')

def turn_off_sr(opt):
    opt.sr = False
    print('Sr has been turned off!')