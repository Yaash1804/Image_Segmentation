import sys
import argparse
from argparse import Namespace
sys.path.append("../")
sys.path.append("./")
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
import torchvision.transforms as transforms

def main():
    args = Namespace(
        data_name='ISIC',
        data_dir="data/ISIC/ISBI2016_ISIC_Part3B_Training_Data",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=5,
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=500,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu=None,
        out_dir='data/ISIC/output',
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=1,
        in_ch=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        dpm_solver=False,
        learn_sigma=True,
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
        version='new',
        use_kl=False,
        predict_xstart=False
    )

    temp = {
        'image_size': 256,
        'num_channels': 128,
        'num_res_blocks': 2,
        'num_heads': 1,
        'in_ch': 4,
        'num_heads_upsample': -1,
        'num_head_channels': -1,
        'attention_resolutions': '16',
        'channel_mult': '',
        'dropout': 0.0,
        'class_cond': False,
        'use_checkpoint': False,
        'use_scale_shift_norm': False,
        'resblock_updown': False,
        'use_fp16': False,
        'use_new_attention_order': False,
        'learn_sigma': True,
        'diffusion_steps': 1000,
        'noise_schedule': 'linear',
        'timestep_respacing': '',
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
        'dpm_solver' : True,
        'version': 'new'
    }

    #dist_util.setup_dist(args)
    #logger.configure(dir=args.out_dir)

    #logger.log("creating data loader...")

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_train = transforms.Compose(tran_list)
        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size, args.image_size))]
        transform_train = transforms.Compose(tran_list)
        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    else:
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory: ", args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True
    )
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(**temp)
    
    if args.multi_gpu:
        model = th.nn.DataParallel(model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device=th.device('cuda', int(args.gpu_dev)))
    '''else:
        model.to(dist_util.dev())'''
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=args.diffusion_steps)

    logger.log("training...")

    train_loop = TrainLoop(
      model=model,
      diffusion=diffusion,
      classifier=None,
      data=data,
      dataloader=datal,
      batch_size=8,  # Default value
      microbatch=2,  # Default value
      lr=0.001,  # Set a default learning rate
      ema_rate=0.999,  # Set a default ema_rate
      log_interval=1,  # Set a default log_interval
      save_interval=500,  # Set a default save_interval
      resume_checkpoint='temp_results/model008000.pt',  # Set to None or another default if applicable
      use_fp16=False,  # Default value
      fp16_scale_growth=1,  # Set a default or pull from args if necessary
      schedule_sampler=schedule_sampler,  # Set to None or another default if applicable
      weight_decay=0.0,  # Set a default weight decay
      lr_anneal_steps=10000,  # Set a default or pull from args if necessary
    )
    train_loop.run_loop()

def create_argparser():
    defaults = dict(
        data_name='BRATS',
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu=None,
        out_dir='results'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
