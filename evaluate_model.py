import argparse
import os
import torch
import time
from attrdict import AttrDict
from thop import profile
import matplotlib.pyplot as plt

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='D:/Desktop/基于深度学习的车辆行为预测研究/Code/sgan-master/scripts/runs/atten'
                                            '/atten_Decoder_before_with_model.pt', type=str)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        attentype=False)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            # 添加计时
            start_time = time.time()

            # plt.figure()

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))
                # 计算推理时间
                # end_time = time.time()
                # inference_time = end_time - start_time
                # print(f'Inference time for batch: {inference_time:.4f} seconds')
                # input = torch.randn(2, 128, 64)

                # flops, params = profile(generator, (obs_traj, obs_traj_rel, seq_start_end))
                # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

            # draw_pred = pred_traj_fake.to('cpu')
            # draw_true = pred_traj_gt.to('cpu')
            # plt.plot(draw_pred[:, :, 0].detach().numpy(), draw_pred[:, :, 1].detach().numpy(), linewidth=1.0,
            #              color='black', label='pred')
            # plt.plot(draw_true[:, :, 0].detach().numpy(), draw_true[:, :, 1].detach().numpy(), linewidth=1.0,
            #              color='red', label='true')
            # # plt.legend()
            # plt.show()

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    # for path in paths:
    #     checkpoint = torch.load(path)
    #     print(path)
    #     for i in checkpoint:
    #         print(i)

    for path in paths:
        checkpoint = torch.load(path)
        # for i in checkpoint:
        #     print(i)
        generator = get_generator(checkpoint)
        # nelement()：统计Tensor的元素个数
        # .parameters()：生成器，迭代的返回模型所有可学习的参数，生成Tensor类型的数据
        total = sum([param.nelement() for param in generator.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))




        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Obs Len: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}'.format(
            _args.dataset_name, _args.obs_len, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
