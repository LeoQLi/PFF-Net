import os, sys
import shutil
import time
import argparse
import numpy as np
import scipy.spatial as spatial
import torch
import torch.nn.functional as F

from net.network import Network
from misc import get_logger, seed_all
from dataset import PointCloudDataset, PatchDataset, SequentialPointcloudPatchSampler, load_data, PCATrans


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='')
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--ckpt_dirs', type=str, default='', help='multiple files separated by comma')
    parser.add_argument('--ckpt_iters', type=str, default='', help='multiple files separated by comma')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--testset_list', type=str, default='')
    parser.add_argument('--eval_list', type=str, nargs='*',
                        help='list of .txt files containing sets of point cloud names for evaluation')
    parser.add_argument('--patch_size', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=0)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--sparse_patch', type=eval, default=True, choices=[True, False],
                        help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--save_pn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--metric', type=str, default='RMSE', choices=['CND', 'RMSE'])
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    test_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='test',
            data_set=args.data_set,
            data_list=args.testset_list,
            sparse_patch=args.sparse_patch,
        )
    test_set = PatchDataset(
            datasets=test_dset,
            patch_size=args.patch_size,
            list_trans=['PCA'],
            sample_size=args.sample_size,
            seed=args.seed,
        )
    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            sampler=SequentialPointcloudPatchSampler(test_set),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    return test_dset, test_dataloader


def normal_RMSE(normal_gts, normal_preds, eval_file='log.txt'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()
        # print(out_str)

    rms = []
    rms_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5 = []
    pgp_alpha = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented RMSE
        ####################################################################
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        ### portion of good points
        rms.append(np.sqrt(np.mean(np.square(ang))))
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))
        pgp_alpha.append(pgp_alpha_shape)

        ### Oriented RMSE
        ####################################################################
        ang_o = np.rad2deg(np.arccos(nn))   # angle error in degree
        ids = ang_o > 90.0
        p = sum(ids) / normal_pred.shape[0]

        ### if more than half of points have wrong orientation, then flip all normals
        if p > 0.5:
            nn = np.sum(np.multiply(normal_gt, -1 * normal_pred), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1
            ang_o = np.rad2deg(np.arccos(nn))    # angle error in degree
            ids = ang_o > 90.0
            p = sum(ids) / normal_pred.shape[0]

        rms_o.append(np.sqrt(np.mean(np.square(ang_o))))

    avg_rms   = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    log_file.close()

    return avg_rms, avg_rms_o


def eval_normal(normal_gt_path, normal_pred_path, output_dir):
    print('\n  Evaluation ...')
    eval_summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(eval_summary_dir, exist_ok=True)

    all_avg_rms = []
    all_avg_rms_o = []
    for cur_list in args.eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, 'list', cur_list + '.txt')
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all shape data of the list
        normal_gts = []
        normal_preds = []
        for shape in shape_names:
            print(shape)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal.npy'))                  # (n, 3)
            normal_gt = load_data(filedir=normal_gt_path, filename=shape + '.normals', dtype=np.float32)  # (N, 3)

            if os.path.exists(os.path.join(normal_gt_path, shape + '.pidx')):
                points_idx = load_data(filedir=normal_gt_path, filename=shape + '.pidx', dtype=np.int32)  # (n,)
            else:
                points_idx = np.arange(normal_gt.shape[0])

            if args.metric == 'CND':
                shape_clean = shape.split('_noise_')[0]
                xyz_used = load_data(filedir=normal_gt_path, filename=shape + '.xyz', dtype=np.float32)
                xyz_clean = load_data(filedir=normal_gt_path, filename=shape_clean + '.xyz', dtype=np.float32)

                sys.setrecursionlimit(int(max(1000, round(xyz_clean.shape[0]/10))))
                kdtree = spatial.cKDTree(xyz_clean, 10)
                qurey_points = xyz_used[points_idx, :]
                _, nor_idx = kdtree.query(qurey_points)

                normal_gt = normal_gt[nor_idx, :]
            elif args.metric == 'RMSE':
                normal_gt = normal_gt[points_idx, :]

            if normal_pred.shape[0] > normal_gt.shape[0]:
                normal_pred = normal_pred[points_idx, :]
            normal_gts.append(normal_gt)
            normal_preds.append(normal_pred)

        ### compute RMSE per-list
        avg_rms, avg_rms_o = normal_RMSE(normal_gts=normal_gts,
                            normal_preds=normal_preds,
                            eval_file=os.path.join(eval_summary_dir, cur_list + '_evaluation_results.txt'))
        all_avg_rms.append(avg_rms)
        all_avg_rms_o.append(avg_rms_o)

        print('### RMSE: %f' % avg_rms)
        print('### RMSE_Ori: %f' % avg_rms_o)

    s = '\n {} \n All RMS unoriented (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_rms), np.mean(all_avg_rms))
    print(s)

    # s = '\n {} \n All RMS oriented (shape average): {} | Mean: {}\n'.format(
    #             normal_pred_path, str(all_avg_rms_o), np.mean(all_avg_rms_o))
    # print(s)

    ### delete the normal files
    if not args.save_pn:
        shutil.rmtree(normal_pred_path)
    return all_avg_rms, all_avg_rms_o


pca = PCATrans()
def PCARotate(coord):
    coord = torch.FloatTensor(coord)
    coord, trans = pca.pca_trans(coord)
    return coord.numpy(), trans.numpy()


def test_patch(ckpt_dir, ckpt_iter):
    ### Input/Output
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pt' % ckpt_iter)
    output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'pred_normal')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(file_save_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=_device)
    model = Network(num_pcl=args.patch_size,
                    num_sam=args.sample_size,
                    encode_knn=args.encode_knn,
                ).to(_device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of trainable parameters: %d' % num_params)

    model.load_state_dict(ckpt)
    model.eval()

    ### get all shape names in the dataset
    data_list = []
    with open(os.path.join(args.dataset_root, args.data_set, 'list', 'testset_%s.txt' % args.data_set)) as f:
        data_list = f.readlines()
    data_list= [x.strip() for x in data_list]
    data_list= list(filter(None, data_list))

    smoothing = 0.9
    total_time = 0.0
    use_pca = True
    ### per-shape prediction
    for idx, item_name in enumerate(data_list):
        shape_time = 0
        coord = load_data(filedir=os.path.join(args.dataset_root, args.data_set), filename=item_name + '.xyz', dtype=np.float32)
        # if args.sparse_patch:
        #     idx_data = load_data(filedir=os.path.join(args.dataset_root, args.data_set), filename=item_name + '.pidx', dtype=np.int32)
        # else:
        idx_data = np.arange(coord.shape[0])

        logger.info('{}/{}: {}, {}'.format(idx+1, len(data_list), coord.shape[0], item_name))
        rand_index = np.random.rand(idx_data.shape[0]) * 1e-3    # random index of points
        idx_uni = np.array([])

        ### generate patch
        l_scale = 4       # determined by the network
        idx_list, coord_list, trans_list  = [], [], []
        kdtree = spatial.cKDTree(coord, 10)

        while idx_uni.size != idx_data.shape[0]:     # looks like a soft farthest cropping. Must cover all index (points)
            query_idx = np.argmin(rand_index)        # randomly select a point with the smallest index

            dists, idx_knn = kdtree.query(coord[query_idx], k=args.patch_size)  # sorted by distance (nearest first)
            coord_pat = coord[idx_knn]                                          # xyz of the kNN

            idx_knn = idx_knn[:args.patch_size//l_scale]                        # only use a subset since the patch size is reduced
            idx_pat_sub = idx_data[idx_knn]

            coord_pat = (coord_pat - coord[query_idx]) / max(dists)

            if use_pca:
                coord_pat, trans_pat = PCARotate(coord_pat)
            else:
                trans_pat = np.eye(3, 3).astype(np.float32)

            idx_list.append(idx_pat_sub), coord_list.append(coord_pat), trans_list.append(trans_pat)
            idx_uni = np.unique(np.concatenate((idx_uni, idx_pat_sub)))

            ### update rand_index
            idx_knn_sub = idx_knn[:int(len(idx_knn) * 0.7)]       # sorted indices of less kNN, eliminate points that are too far from the selected point
            dists_sub = dists[:len(idx_knn_sub)]
            delta = np.square(1 - dists_sub / np.max(dists_sub))  # far points have smaller delta
            rand_index[idx_knn_sub] += delta                      # increment the random index, so that next index will be quite far from this one.

        ### inference
        pred = torch.zeros(coord.shape).cuda()
        batch_num = int(np.ceil(len(idx_list) / args.batch_size))
        for i in range(batch_num):
            s_i = i * args.batch_size
            e_i = min((i + 1) * args.batch_size, len(idx_list))
            idx_part, coord_part, trans_part = idx_list[s_i:e_i], coord_list[s_i:e_i], trans_list[s_i:e_i]

            idx_part = np.stack(idx_part, 0)  # (B, N')
            coord_part = torch.FloatTensor(np.stack(coord_part, 0)).cuda(non_blocking=True) # (B, N, 3)
            trans_part = torch.FloatTensor(np.stack(trans_part, 0)).cuda(non_blocking=True) # (B, 3, 3)

            start_time = time.time()
            with torch.no_grad():
                _, pred_part = model(coord_part, mode_test=True)    # (B, N', 3)
            end_time = time.time()
            elapsed_time = end_time - start_time
            shape_time += elapsed_time
            total_time += elapsed_time

            if use_pca:
                pred_part = torch.bmm(pred_part, trans_part.transpose(2, 1))  # (B, N', 3)

            for idx_update, pred_update in zip(idx_part, pred_part):
                pred_temp = torch.where((pred[idx_update, :] - pred_update).pow(2).sum(1, keepdim=True) -
                                        (pred[idx_update, :] + pred_update).pow(2).sum(1, keepdim=True) > 0,
                                    -pred_update, pred_update)
                pred[idx_update, :] = smoothing * pred[idx_update, :] + (1 - smoothing) * pred_temp
            logger.info('Test: {}/{}, Number of output normals: {}, Current batch size: {}'.format(e_i, len(idx_list), idx_part.shape[1], idx_part.shape[0]))

        pred = F.normalize(pred, p=2, dim=-1)
        pred = pred.data.cpu().numpy()

        logger.info(f'Shape takes {shape_time:.3f} sec. Batch size is {args.batch_size}. Number of input point is {args.patch_size}')

        pred_save_path = os.path.join(file_save_dir, '{}_normal.npy'.format(item_name))
        np.save(pred_save_path, pred)
    logger.info(f'Takes {total_time:.3f} sec for {args.data_set}, ({total_time/len(data_list):.3f} sec per shape.)')
    return output_dir, file_save_dir


def test_point(ckpt_dir, ckpt_iter):
    test_dset, test_dataloader = get_data_loaders(args)

    ### Input/Output
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pt' % ckpt_iter)
    output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'pred_normal')
    os.makedirs(file_save_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=_device)
    model = Network(encode_knn=args.encode_knn).to(_device)

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_params = sum([np.prod(p.size()) for p in model_parameters])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of trainable parameters: %d' % num_params)

    model.load_state_dict(ckpt)
    model.eval()

    shape_ind = 0
    shape_patch_offset = 0
    shape_num = len(test_dset.shape_names)
    shape_patch_count = test_dset.shape_patch_count[shape_ind]

    num_batch = len(test_dataloader)
    normal_prop = torch.zeros([shape_patch_count, 3])

    total_time = 0.0
    for batchind, batch in enumerate(test_dataloader, 0):
        pcl_pat = batch['pcl_pat'].to(_device)
        data_trans = batch['pca_trans'].to(_device) if 'pca_trans' in batch else None

        start_time = time.time()
        with torch.no_grad():
            n_est, nn_est = model(pcl_pat, mode_test=True)
            # n_est = nn_est[:,0,:]
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        if batchind % 5 == 0:
            batchSize = pcl_pat.size()[0]
            logger.info('[%d/%d] %s: time per patch: %.3f ms' % (
                        batchind, num_batch-1, test_dset.shape_names[shape_ind], 1000 * elapsed_time / batchSize))

        if data_trans is not None:
            ### transform predictions with inverse pca rotation (back to world space)
            n_est = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

        ### Save estimated normals to file
        batch_offset = 0
        while batch_offset < n_est.shape[0] and shape_ind + 1 <= shape_num:
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset

            ### append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

            if shape_patches_remaining <= batch_patches_remaining:
                normal_prop = F.normalize(normal_prop, p=2, dim=-1)
                normals_to_write = normal_prop.cpu().numpy()
                # eps=1e-6
                # normals_to_write[np.logical_and(normals_to_write < eps, normals_to_write > -eps)] = 0.0

                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal.npy') # for faster reading speed
                np.save(save_path, normals_to_write)

                logger.info('Save normal: {}'.format(save_path))
                logger.info('Total Time: %.2f sec, Shape Num: %d / %d \n' % (total_time, shape_ind+1, shape_num))

                sys.stdout.flush()
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < shape_num:
                    shape_patch_count = test_dset.shape_patch_count[shape_ind]
                    normal_prop = torch.zeros([shape_patch_count, 3])

    logger.info('Total Time: %.2f sec, Time per shape: %.2f = %.2f / %d' % (total_time, total_time/shape_num, total_time, shape_num))
    return output_dir, file_save_dir



if __name__ == '__main__':
    # Arguments
    args = parse_arguments()
    arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
    print('Arguments:\n %s\n' % arg_str)

    seed_all(args.seed)
    PID = os.getpid()

    assert args.gpu >= 0, "ERROR GPU ID!"
    _device = torch.device('cuda:%d' % args.gpu)

    ckpt_dirs = args.ckpt_dirs.split(',')
    ckpt_iters = args.ckpt_iters.split(',')

    for ckpt_dir in ckpt_dirs:
        eval_dict = ''
        sum_file = 'eval_' + args.data_set + ('_'+args.tag if len(args.tag) != 0 else '')
        log_file_sum = open(os.path.join(args.log_root, ckpt_dir, sum_file+'.txt'), 'a')
        log_file_sum.write('\n====== %s ======\n' % args.eval_list)

        for ckpt_iter in ckpt_iters:
            output_dir, file_save_dir = test_point(ckpt_dir=ckpt_dir, ckpt_iter=ckpt_iter)
            if not output_dir or args.data_set in ['Semantic3D', 'KITTI_sub', 'WireframePC', 'Others']:
                continue
            all_avg_rms, all_avg_rms_o = eval_normal(normal_gt_path=os.path.join(args.dataset_root, args.data_set),
                                                normal_pred_path=file_save_dir,
                                                output_dir=output_dir)

            s = '%s: %s | Mean: %f \t|| %s | Mean: %f\n' % (ckpt_iter, str(all_avg_rms), np.mean(all_avg_rms),
                                                                    str(all_avg_rms_o), np.mean(all_avg_rms_o))
            log_file_sum.write(s)
            log_file_sum.flush()
            eval_dict += s

        log_file_sum.close()
        s = '\n All RMS not oriented and oriented (shape average): \n{}\n'.format(eval_dict)
        print(s)


