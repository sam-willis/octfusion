import os
from pathlib import Path
import time
import inspect
import random

from termcolor import colored, cprint
from tqdm import tqdm

import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
from datetime import datetime

import trimesh
from options.train_options import TrainOptions
from datasets.dataloader import config_dataloader, get_data_generator
from models.base_model import create_model

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from utils.util import seed_everything, category_5_to_label, category_5_to_num

import torch
from utils.visualizer import Visualizer
import openvsp as vsp

def train_main_worker(opt, model, train_loader, test_loader, visualizer):

    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    train_dg = get_data_generator(train_loader)
    test_dg = get_data_generator(test_loader)

    epoch_length = len(train_loader)
    print('The epoch length is', epoch_length)

    total_iters = epoch_length * opt.epochs
    start_iter = opt.start_iter

    epoch = start_iter // epoch_length

    # pbar = tqdm(total=total_iters)
    pbar = tqdm(range(start_iter, total_iters))

    iter_start_time = time.time()
    for iter_i in range(start_iter, total_iters):

        opt.iter_i = iter_i
        iter_ip1 = iter_i + 1

        if get_rank() == 0:
            visualizer.reset()

        data = next(train_dg)
        data['iter_num'] = iter_i
        data['epoch'] = epoch
        model.set_input(data)
        model.optimize_parameters()

        # if torch.isnan(model.loss).any() == True:
        #     break

        if get_rank() == 0:
            pbar.update(1)
            if iter_i % opt.print_freq == 0:
                errors = model.get_current_errors()

                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(iter_i, errors, t)

            if iter_ip1 % opt.save_latest_freq == 0:
                cprint('saving the latest model (current_iter %d)' % (iter_i), 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)

            # save every 3000 steps (batches)
            if iter_ip1 % opt.save_steps_freq == 0:
                cprint('saving the model at iters %d' % iter_ip1, 'blue')
                latest_name = f'steps-latest'
                model.save(latest_name, iter_ip1)
                cur_name = f'steps-{iter_ip1}'
                model.save(cur_name, iter_ip1)

                cprint(f'[*] End of steps %d \t Time Taken: %d sec \n%s' %
                    (
                        iter_ip1,
                        time.time() - iter_start_time,
                        os.path.abspath(os.path.join(opt.logs_dir, opt.name))
                    ), 'blue', attrs=['bold']
                )

            if iter_i % epoch_length == epoch_length - 1:
                print('Finish One Epoch!')
                epoch += 1
                print('Now Epoch is:', epoch)

        # display every n batches
        if iter_i % opt.display_freq == 0:
            if iter_i == 0 and opt.debug == "0":
                pbar.update(1)
                continue

            # eval
            if opt.model == "vae":
                data = next(test_dg)
                data['iter_num'] = iter_i
                data['epoch'] = epoch
                model.set_input(data)
                model.inference(save_folder = f'temp/{iter_i}')
            else:
                if opt.category == "im_5":
                    category = random.choice(list(category_5_to_num.keys()))
                else:
                    category = opt.category
                
                model.sample(category = category, prefix = 'results', ema = True, ddim_steps = 200, save_index = iter_i)
            
            # torch.cuda.empty_cache()

        if opt.update_learning_rate:
            model.update_learning_rate_cos(epoch, opt)

        

def generate_vae(opt, model, test_loader):
    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    test_dg = get_data_generator(test_loader)

    epoch_length = len(train_loader)
    print('The epoch length is', epoch_length)

    total_iters = epoch_length
    start_iter = 0

    # pbar = tqdm(total=total_iters)
    pbar = tqdm(range(start_iter, total_iters))

    for iter_i in range(start_iter, total_iters):

        data = next(test_dg)
        data['iter_num'] = iter_i
        data['epoch'] = 0
        model.set_input(data)
        seed_everything(opt.seed)
        model.inference()
        pbar.update


def generate(opt, model, index = 0):
    # why is there a limit? This is a bit sus.
    # total_num = category_5_to_num[opt.category]
    result_index = index * get_world_size() + get_rank()
    assert opt.split_dir is None 
    split_small = None
    model.batch_size = 1
    assert opt.category != "im_5"
    category = opt.category

    model.sample(
        split_small=split_small, 
        category=category, 
        prefix='results', 
        ema=True, 
        ddim_steps=200, 
        clean=False, 
        save_index=result_index
    )
    # this is a horrible implicit dep
    mesh_dir = Path(opt.logs_dir) / opt.name / f"results_{category}"
    for mesh_fn in (p for p in mesh_dir.iterdir() if p.is_file() and p.suffix == ".obj"):
        mesh = trimesh.load(mesh_fn, force='mesh')
        mesh.export(mesh_fn.with_suffix(".stl") , file_type='stl')
        # mesh.is_watertight
        # print(mesh.volume / mesh.convex_hull.volume)
        # do something with openvsp or openfoam

    
if __name__ == "__main__":
    # this will parse args, setup log_dirs, multi-gpus
    opt = TrainOptions().parse_and_setup()
    device = opt.device
    rank = opt.rank
    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')      

    # main loop
    model = create_model(opt)
    opt.start_iter = model.start_iter
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    assert rank == 0, "Only support single GPU training now."

    # visualizer
    visualizer = Visualizer(opt)
    visualizer.setup_io()
    expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
    model_f = inspect.getfile(model.__class__)
    modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
    os.system(f'cp {model_f} {modelf_out}')
    dset_f = "datasets/dualoctree_snet.py"
    dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
    os.system(f'cp {dset_f} {dsetf_out}')
    sh_f = 'scripts/run_snet_uncond.sh'
    sh_out = os.path.join(expr_dir, os.path.basename(sh_f))
    os.system(f'cp {sh_f} {sh_out}')
    train_f = 'train.py'
    train_out = os.path.join(expr_dir, os.path.basename(train_f))
    os.system(f'cp {train_f} {train_out}')
        
    if opt.vq_cfg is not None:
        vq_cfg = opt.vq_cfg
        cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
        os.system(f'cp {vq_cfg} {cfg_out}')

    if opt.df_cfg is not None:
        df_cfg = opt.df_cfg
        cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
        os.system(f'cp {df_cfg} {cfg_out}')
    
    generate(opt, model)
