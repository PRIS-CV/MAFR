import time
import torchvision.transforms as transforms
from torch.distributions import Categorical
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import *
from network import *
from configs import *

import math
import argparse

import models.resnet as resnet
import models.densenet as densenet
from models import create_model

import numpy as np
import torch
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='Training code')

parser.add_argument('--data_url', default='../datasets/MvCars-V2-Lite/', type=str,
                    help='path to the dataset (ImageNet)')

parser.add_argument('--class_num', default=233, type=int,
                    help='class number of dataset')

parser.add_argument('--view_num', default=8, type=int,
                    help='view number of dataset')

parser.add_argument('--work_dirs', default='./output', type=str,
                    help='path to save log and checkpoints')

parser.add_argument('--fc_lr', default=None, type=float,
                    help='learning rate for full layer')

parser.add_argument('--fc_num', default=1, type=int,
                    help='1 for shared fc and actor for each step and \
                    step number for independent fc and actor for each step')

parser.add_argument('--ppo_lr', default=5e-3, type=float,
                    help='learning rate for PPO')

parser.add_argument('--train_stage', default=-1, type=int,
                    help='select training stage, see our paper for details \
                          stage-1 : warm-up \
                          stage-2 : learn to select patches with RL \
                          stage-3 : finetune CNNs')

parser.add_argument('--test_stage', default=False, type=bool,
                    help='test the model')

parser.add_argument('--model_arch', default='resnet50', type=str,
                    help='architecture of the model to be trained')

parser.add_argument('--T', default=8, type=int,
                    help='maximum length of the sequence of Glance + Focus')

parser.add_argument('--model_path', default='', type=str,
                    help='path to the pre-trained model of Local Encoder (for training stage-1)')

parser.add_argument('--checkpoint_path', default='', type=str,
                    help='path to the stage-2/3 checkpoint (for training stage-2/3)')

parser.add_argument('--coarse_checkpoint_path', default='', type=str,
                    help='path to the coarse training full layer checkpoint (for training stage-2 loading PPO full layer)')

parser.add_argument('--resume', default='', type=str,
                    help='path to the checkpoint for resuming')

parser.add_argument('--fc_rnn', default='gru', type=str,
                    help='gru, lstm or nfc for classifier')

parser.add_argument('--statistic', default=True, type=bool,
                    help='statistic of the selecting actions')

parser.add_argument('--overlap_mask', default=True, type=bool,
                    help='add mask to avoid repeated actions')

parser.add_argument('--ppo_rewards', default='simple', type=str,
                    help='simple, multidim or randmean')

parser.add_argument('--ppo_fc', default=0., type=float,
                    help='train state_encoder - gru - fc in stage 2 ')

parser.add_argument('--ppo_load_fc', default=False, type=bool,
                    help='load stage 1 full layer parameters in stage 2 ')

parser.add_argument('--current_rewards', default=False, type=bool,
                    help='consider the influence of current action of future rewards or not')

parser.add_argument('--overlap_rewards', default=False, type=bool,
                    help='change the rewards of overlap actions to -rewards.std')

parser.add_argument('--perceptual', default=0., type=float,
                    help='perceptual loss of states step by step in stage 2, to choose visual different views')

parser.add_argument('--same_gru', default=False, type=bool,
                    help='PPO without state encoder')

parser.add_argument('--coarse_training', default=False, type=bool,
                    help='train coarse stage 1 full layer for stage 2 to load')

parser.add_argument('--temperature', default=1, type=int,
                    help='make output distribution smoother')

parser.add_argument('--entropy', default=0., type=float,
                    help='add negative entropy loss to make output distribution smoother')

parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')

args = parser.parse_args()


def main():
    if args.seed is not None:
        seed_torch(args.seed)

    # *create model* #
    model_configuration = model_configurations[args.model_arch]
    if 'resnet' in args.model_arch:
        model_arch = 'resnet'
        model = resnet.resnet50(pretrained=False)

    train_configuration = train_configurations[model_arch]

    if args.coarse_training:
        fc = Full_layer(model_configuration['feature_num'],
                    model_configuration['fc_hidden_dim'],
                    args.fc_rnn, 4, train_configuration['batch_size'])
    else:
        fc = Full_layer(model_configuration['feature_num'],
                    model_configuration['fc_hidden_dim'],
                    args.fc_num, args.fc_rnn, args.class_num, train_configuration['batch_size'])

    if args.train_stage == 1 and args.checkpoint_path != None:
        if args.model_path:
            state_dict = torch.load(args.model_path)
            model.load_state_dict(state_dict)
    elif args.coarse_training:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(args.checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']
        if args.coarse_checkpoint_path:
            coarse_checkpoint = torch.load(args.coarse_checkpoint_path)
        model.load_state_dict(model_state_dict)
        fc.load_state_dict(checkpoint['fc'])
    
    if not os.path.isdir(args.work_dirs):
        mkdir_p(args.work_dirs)

    if args.train_stage == 1:
        record_path = args.work_dirs \
                  + '/train-stage' + str(args.train_stage) \
                #   + '_temperature' + str(args.temperature) \
                #   + '_coarse-' + str(args.coarse_training) \
                #   + '_entropy' + str(args.entropy) \
                #   + '_batch-size' + str(train_configuration['batch_size']) \
                #   + '_epoch' + str(train_configuration['epoch_num'])
    elif args.train_stage == 2:
        record_path = args.work_dirs \
                  + '/train-stage' + str(args.train_stage) \
                #   + '_ppo-lr' + str(args.ppo_lr)
                #   + '_checkpoint-' + str(args.checkpoint_path.split('/')[1].split('_')[1]) \
                #   + '_nocurrent'
                #   + '_ppo-rewards-' + str(args.ppo_rewards) \
                #   + '_current-rewards-' + str(args.current_rewards) \
                #   + '-' + str(args.checkpoint_path.split('/')[1].split('_')[2]) \
                #   + '_batch-size' + str(train_configuration['batch_size']) 
    else:
        record_path = args.work_dirs \
                  + '/train-stage' + str(args.train_stage) \
                #   + '_' + str(args.checkpoint_path.split('/')[1].split('_')[1]) \
                #   + '_nocurrent'
                #   + '-' + str(args.checkpoint_path.split('/')[1].split('_')[2]) \

    if not os.path.isdir(record_path):
        mkdir_p(record_path)
    record_file = record_path + '/record.txt'
    
    if args.train_stage != 2:
        optimizer = torch.optim.SGD([{'params': model.parameters()},
                                     {'params': fc.parameters()}],
                                    lr=0,  # specify in adjust_learning_rate()
                                    momentum=train_configuration['momentum'],
                                    nesterov=train_configuration['Nesterov'],
                                    weight_decay=train_configuration['weight_decay'])
        training_epoch_num = train_configuration['epoch_num']
    else:
        optimizer = None
        training_epoch_num = 15
    criterion = nn.CrossEntropyLoss().cuda()

    model = nn.DataParallel(model.cuda())
    fc = fc.cuda()

    traindir = args.data_url + 'train/'
    valdir = args.data_url + 'val/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    if args.coarse_training:
        train_set = MultiViewCoarseDataset(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    # transforms.Resize((224,224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize]))
        train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=train_configuration['batch_size'], num_workers=8, pin_memory=False,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(torch.randperm(len(train_set))))
        
        val_set = MultiViewCoarseDataset(valdir, transforms.Compose([
                    transforms.Resize(256),
                    # transforms.Resize((224,224)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize]))
        val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=train_configuration['batch_size'], shuffle=False, num_workers=8, pin_memory=False)
    else:
        if args.train_stage != 2:
            train_set = MultiViewDataset(traindir, transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop(224),
                        transforms.ToTensor(),
                        normalize]), args)
        else:
            train_set = MultiViewDataset(valdir, transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize]), args)
        train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=train_configuration['batch_size'], num_workers=8, pin_memory=False,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(torch.randperm(len(train_set))))
        
        val_set = MultiViewDataset(valdir, transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize]), args)
        val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=train_configuration['batch_size'], shuffle=False, num_workers=8, pin_memory=False)

    if args.train_stage != 1:
        state_dim = model_configuration['feature_map_channels'] * math.ceil(224 / 32) * math.ceil(224 / 32)
        ppo = PPO(model_configuration['feature_map_channels'], state_dim, model_configuration['policy_hidden_dim'], args.view_num, args.fc_num, model_configuration['policy_conv'], args.same_gru, args.coarse_checkpoint_path, train_configuration['batch_size'], args.ppo_lr)
        if args.train_stage == 2 and args.ppo_load_fc:
            ppo.policy.fc.load_state_dict(checkpoint['fc'])
            ppo.policy_old.fc.load_state_dict(checkpoint['fc'])
        elif args.train_stage == 2 and args.coarse_checkpoint_path:
            ppo.policy.fc.load_state_dict(coarse_checkpoint['fc'])
            ppo.policy_old.fc.load_state_dict(coarse_checkpoint['fc'])
        elif args.train_stage == 3:
            ppo.policy.load_state_dict(checkpoint['policy'])
            ppo.policy_old.load_state_dict(checkpoint['policy'])
    else:
        ppo = None
    memory = Memory()

    if args.resume:
        resume_ckp = torch.load(args.resume)

        start_epoch = resume_ckp['epoch']
        print('resume from epoch: {}'.format(start_epoch))

        model.module.load_state_dict(resume_ckp['model_state_dict'])
        fc.load_state_dict(resume_ckp['fc'])

        if optimizer:
            optimizer.load_state_dict(resume_ckp['optimizer'])

        if ppo:
            ppo.policy.load_state_dict(resume_ckp['policy'])
            ppo.policy_old.load_state_dict(resume_ckp['policy'])
            ppo.optimizer.load_state_dict(resume_ckp['ppo_optimizer'])

        best_acc = resume_ckp['best_acc']
        last_acc = resume_ckp['acc']
    else:
        start_epoch = 0
        best_acc = torch.zeros(args.T)

    fd = open(record_file, 'a+')
    print(args)
    fd.write(str(args)+'\n')

    if not args.test_stage:
        for epoch in range(start_epoch, training_epoch_num):
            if args.train_stage != 2:
                print('Training Stage: {}, lr:'.format(args.train_stage))
                adjust_learning_rate(optimizer, train_configuration,
                                    epoch, training_epoch_num, args)
            else:
                print('Training Stage: {}, train ppo only'.format(args.train_stage))

            train(model, fc, memory, ppo, optimizer, train_loader, criterion,
                epoch, train_configuration['batch_size'], record_file, args)
            
            # if (epoch + 1) % 5 == 0:
            accs = []
            for val_time in range(args.view_num):
                acc = validate(model, fc, memory, ppo, optimizer, val_loader, criterion,
                            val_time, epoch, train_configuration['batch_size'], record_file, args)
                print('Val Times Accuracy: [{}/{}]:'.format(val_time, args.view_num))
                print(acc)
                fd.write('Val Times Accuracy: [{}/{}]:\n'.format(val_time, args.view_num))
                fd.write(str(acc) + '\n')
                accs.append(acc)
            acc_mean = torch.Tensor(accs).mean(0)

            # if args.train_stage != 2:
            #     if acc_mean[args.T - 1] > best_acc[args.T - 1]:
            #         best_acc = acc_mean
            #         is_best = True
            #     else:
            #         is_best = False 
            # else:
            if torch.Tensor(acc_mean).mean() > torch.Tensor(best_acc).mean():
                best_acc = acc_mean
                is_best = True
            else:
                is_best = False 

            if is_best:
                print('Epoch [{}]: Best Accuracy!!!!!!'.format(epoch))
                fd.write('Epoch [{}]: Best Accuracy!!!!!!'.format(epoch))
            print('\naverage accuracy of {} validation times:'.format(args.view_num))
            print(acc_mean)
            print()
            fd.write('\naverage accuracy of {} validation times:\n'.format(args.view_num))
            fd.write(str(acc_mean) + '\n')
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'fc': fc.state_dict(),
                'acc': acc_mean,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict() if optimizer else None,
                'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
                'policy': ppo.policy.state_dict() if ppo else None,
            }, is_best, checkpoint=record_path)
    
    else:
        accs = []
        for val_time in range(args.view_num):
            acc = validate(model, fc, memory, ppo, optimizer, val_loader, criterion,
                        val_time, training_epoch_num, train_configuration['batch_size'], record_file, args)
            print('Val Times Accuracy: [{}/{}]:'.format(val_time, args.view_num))
            print(acc)
            fd.write('Val Times Accuracy: [{}/{}]:\n'.format(val_time, args.view_num))
            fd.write(str(acc) + '\n')
            accs.append(acc)
        acc_mean = torch.Tensor(accs).mean(0)

        print('\ntest accuracy!!!!!!')
        print(acc_mean)
        fd.write('\ntest accuracy!!!!!!\n')
        fd.write(str(acc_mean) + '\n')

        print('\nlast accuracy!!!!!!')
        print(last_acc)
        fd.write('\nlast accuracy!!!!!!\n')
        fd.write(str(last_acc) + '\n')

    print('\nbest accuracy!!!!!!')
    print(best_acc)
    fd.write('\nbest accuracy!!!!!!\n')
    fd.write(str(best_acc) + '\n')

    fd.close()

def train(model, fc, memory, ppo, optimizer, train_loader, criterion,
          epoch, batch_size, record_file, args):

    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]

    train_batches_num = len(train_loader)

    if args.coarse_training:
        model.eval()
        fc.train()
    elif args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()

    fd = open(record_file, 'a+')

    end = time.time()

    for iteration, (x, target) in enumerate(train_loader): # x - view_num * batch_size * 3(RGB) * img_size * img_size
        if  x[0].size(0) == batch_size:
            loss_cla, loss_entro, loss_t = [], [], []
            target_var = target.cuda()
            
            if args.train_stage == 1:
                action_nooverlap = torch.zeros(batch_size, args.T)
                for i in range(batch_size):
                    action_nooverlap[i, :] = torch.randperm(args.T)

            temperature_sequence = [5.0, 3.0, 2.0, 1.5, 1.25, 1.125, 1.0625]

            for step in range(args.T):
                if args.train_stage == 1:
                    action = action_nooverlap[:, step].long().cuda()
                elif step == 0:
                    action = torch.randint(len(x), (batch_size, )).cuda()
                else:
                    if step == 1:
                        action = ppo.select_action(state.to(0), memory, args, step, restart_batch = True)
                    else:
                        action = ppo.select_action(state.to(0), memory, args, step)
                
                views = get_view(x, action).cuda()
                
                if args.train_stage != 2:
                    output, state = model(views)
                    if step == 0:
                        _, output = fc(output, step, restart=True)
                    else:
                        _, output = fc(output, step, restart=False)
                else:
                    with torch.no_grad():
                        output, state = model(views)
                        if step == 0:
                            _, output = fc(output, step, restart=True)
                        else:
                            _, output = fc(output, step, restart=False)

                if args.temperature == 0:
                    temperature = temperature_sequence[step]
                else:
                    temperature = args.temperature

                output = output[0]
                loss = criterion(output, target_var)
                loss_cla.append(loss)
                loss_entro.append(Categorical(probs = F.softmax(output, 1)).entropy())
                loss_t.append(F.mse_loss(F.softmax(output, 1), F.softmax(output.detach() / temperature, 1)))
                losses[step].update(loss.data.item(), x[0].size(0))

                acc = accuracy(output, target_var, topk=(1,))
                top1[step].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x[0].size(0))

                confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1)
                
                if args.ppo_rewards == 'randmean':
                    confidence_randmean = torch.zeros(x[0].size(0), len(x))
                    for i in range(len(x)):
                        action = torch.randint(len(x), (x[0].size(0), )).cuda()
                        views = get_view(x, action).cuda()
                        with torch.no_grad():
                            output, state = model(views)
                        confidence_randmean[:,i] = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1)
                    confidence_randmean = confidence_randmean.mean(1).cuda()

                if step != 0:
                    if args.ppo_rewards == 'simple':
                        reward = confidence - confidence_last
                        memory.rewards.append(reward.data)
                    elif args.ppo_rewards == 'randmean':
                        reward = confidence - confidence_randmean
                        memory.rewards.append(reward.data)
                    elif args.ppo_rewards == 'multidim':
                        multi_rewards = torch.zeros(x[0].size(0), len(x))
                        for i in range(len(x)):
                            action = (torch.zeros(x[0].size(0), ) + i).cuda()
                            views = get_view(x, action).cuda()
                            with torch.no_grad():
                                output, state = model(views)
                            multi_rewards[:,i] = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1) - confidence_last
                        memory.rewards.append(multi_rewards.data)
                
                confidence_last = confidence

            loss_total = sum(loss_cla) / args.T + sum(loss_t) / args.T - args.entropy * (sum(loss_entro) / args.T).mean()

            if args.train_stage == 1:
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
            elif args.train_stage == 2:
                if args.ppo_rewards == 'multidim':
                    ppo.multi_update(memory, target_var, criterion, args)
                else:
                    ppo.update(memory, target_var, criterion, args)
            elif args.train_stage == 3:
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
#                 if args.ppo_rewards == 'multidim':
#                     ppo.multi_update(memory, target_var, criterion, args)
#                 else:
#                     ppo.update(memory, target_var, criterion, args)

            memory.clear_memory()

            batch_time.update(time.time() - end)
            end = time.time()

            string = ('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                        'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                epoch, iteration + 1, train_batches_num, batch_time=batch_time, loss=losses[-1]))
            print(string)
            fd.write(string + '\n')

    fd.close()


def validate(model, fc, memory, ppo, _, val_loader, criterion,
             val_time, epoch, batch_size, record_file, args):
    
    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]

    val_batches_num = len(val_loader)

    model.eval()
    fc.eval()

    fd = open(record_file, 'a+')
    
    end = time.time()
    
    view_index = []
    
    trajectory_l = []
    target_l = []

    for iteration, (x, target) in enumerate(val_loader):
        if  x[0].size(0) == batch_size:
            loss_cla = []
            target_var = target.cuda()
            
            if args.train_stage == 1:
                action_nooverlap = torch.zeros(batch_size, args.T-1)
                for i in range(batch_size):
                    all_action = torch.randperm(args.T)
                    action_nooverlap[i, :] = all_action[~np.isin(all_action, val_time)]

            for step in range(args.T):
                if  step == 0:
                    action = (torch.zeros(batch_size, ) + val_time).long().cuda()
                elif args.train_stage == 1:
                    action = action_nooverlap[:, step - 1].long().cuda()
                else:
                    if step == 1:
                        action, top1_action = ppo.select_action(state.to(0), memory, args, step, restart_batch=True, training=False)
                    else:
                        action, top1_action = ppo.select_action(state.to(0), memory, args, step, training=False)

                if args.statistic:
                    if args.train_stage == 1 or step == 0:
                        view_index += action.cpu().numpy().tolist()
                    else:
                        view_index += top1_action.cpu().numpy().tolist()
                
                views = get_view(x, action).cuda()
                with torch.no_grad():
                    output, state = model(views)

                if step == 0:
                    _, output = fc(output, step, restart=True)
                else:
                    _, output = fc(output, step, restart=False)

                output = output[0]
                loss = criterion(output, target_var)
                loss_cla.append(loss)
                losses[step].update(loss.data.item(), x[0].size(0))

                acc = accuracy(output, target_var, topk=(1,))
                top1[step].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x[0].size(0))

                confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1)
                
#                 print(step, F.softmax(output.detach(), 1).max(1)[0])
                if step != 0:
                    reward = confidence - confidence_last
                    memory.rewards.append(reward)

                confidence_last = confidence
                memory.actions.append(action)
            
            trajectory = torch.cat([act.unsqueeze(1) for act in memory.actions], 1)
            trajectory_l.append(trajectory)
            target_l.append(target)
            
            memory.clear_memory()

            batch_time.update(time.time() - end)
            end = time.time()

            string = ('Val: [{0}][{1}/{2}]\t'
                        'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                        'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                epoch, iteration + 1, val_batches_num, batch_time=batch_time, loss=losses[-1]))
            print(string)
            fd.write(string + '\n')
    
#     trajectory_l = np.array(torch.cat(trajectory_l, 0).cpu())
#     target_l = np.array(torch.cat(target_l, 0).cpu())
#     np.save("for_vis/trajectory_step-{}.npy".format(val_time), trajectory_l)
#     np.save("for_vis/target_step-{}.npy".format(val_time), target_l)
    
    if args.statistic:
        view_dict = {}
        for index in view_index:
            view_dict[index] = view_dict.get(index, 0) + 1

        print('statistic of frequency of appearing actions:')
        print(view_dict)
        fd.write('statistic of frequency of appearing actions:\n')
        fd.write(str(view_dict) + '\n')

    fd.close()

    # return [round(top1[i].ave, 2) for i in range(args.T)]
    return [top1[i].ave for i in range(args.T)]



if __name__ == '__main__':
    main()