import torch
from torch.utils.data import DataLoader
import sys
import time
import os

from pair_input import InputPair
from siamfc_tracker import TrackerSiamFC


# load the newest checkpoint to resume training.
def resume_train():
    resume_path = 'saved/training_resume_state/'
    tars = os.listdir(resume_path)
    tars_flag = [(int(a.split('_')[1]), int(a.split('_')[3])) for a in tars if len(a.split('_')) == 5]

    if len(tars_flag) == 0:
        print('no train checkpoint loaded in', resume_path)
        return None
    else:
        tars_flag.sort()
        newest_epoch, newest_step = tars_flag[-1]
        tar_name = 'epoch_{}_step_{}_.tar'.format(newest_epoch, newest_step)

        checkpoint = torch.load(resume_path + tar_name)
        print('load newest train checkpoint:', tar_name)

        return checkpoint


# save for resuming training
def save_train(epoch, step, model_state_dict, optimizer_state_dict, lr_scheduler_state_dict):
    save_path = 'saved/training_resume_state/epoch_{}_step_{}_.tar'.format(epoch, step)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_scheduler_state_dict
               },
               save_path)
    print('save train checkpoint in', save_path)


# save for test
def save_weights(epoch, step, state_dict):
    save_path = 'saved/test_model_weights/epoch_{}_step_{}_.pth'.format(epoch, step)
    torch.save(state_dict, save_path)
    print('save test weights in', save_path)


# batch per GPU
batch_size = 16

# used in dataloader
workers_num = 8
GPU_num = 2

if __name__ == '__main__':
    input_pair = InputPair('data')

    cuda = torch.cuda.is_available()

    if cuda:
        print('GPU is available!')
    else:
        print('GPU is not available!')

    tracker = TrackerSiamFC(mode='train')

    input_batch = DataLoader(input_pair, batch_size=batch_size*GPU_num, shuffle=True,
                             pin_memory=cuda, drop_last=True, num_workers=workers_num)

    start_epoch = 0
    checkpoint = resume_train()
    if checkpoint is not None:
        tracker.model.load_state_dict(checkpoint['model_state_dict'])
        tracker.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tracker.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        start_epoch = checkpoint['epoch']

    # log setting
    log_batch_size = 20
    log_batch_time = time.time()

    try:
        # training loop
        max_epoch = 50
        for epoch in range(start_epoch, max_epoch):

            # show learning rate
            for param_group in tracker.optimizer.param_groups:
                print('learning rate in previous epoch', param_group['lr'])

            for step, batch in enumerate(input_batch):

                loss = tracker.train_step(
                    batch,update_lr=(step == 0))

                if step % log_batch_size == 0:
                    log_batch_time = time.time() - log_batch_time

                    print('Epoch {:^3}({:5}/{}) | Loss: {:.3f} | Time per batch: {:.2f}s'.format(
                        epoch + 1,
                        step + 1,
                        len(input_batch),
                        loss,
                        log_batch_time/log_batch_size))

                    sys.stdout.flush()

                    log_batch_time = time.time()

            if epoch % 10 == 9:
                save_train(epoch, step,
                           tracker.model.state_dict(),
                           tracker.optimizer.state_dict(),
                           tracker.lr_scheduler.state_dict())
            if epoch == max_epoch - 1:
                if isinstance(tracker.model, torch.nn.DataParallel):
                    tracker.model = tracker.model.module
                save_weights(epoch, step, tracker.model.state_dict())
    except:
        if step > 0:
            save_train(epoch, step,
                       tracker.model.state_dict(),
                       tracker.optimizer.state_dict(),
                       tracker.lr_scheduler.state_dict())

            if isinstance(tracker.model, torch.nn.DataParallel):
                tracker.model = tracker.model.module
            save_weights(epoch, step, tracker.model.state_dict())

