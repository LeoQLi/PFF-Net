import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='', choices=['train', 'test'])
parser.add_argument('--dataset_root', type=str, default='/data/Dataset/')
parser.add_argument('--log_root', type=str, default='./log')
parser.add_argument('--data_set', type=str, default='PCPNet',
                    choices=['PCPNet', 'SceneNN', 'ScanNet', 'Semantic3D', 'KITTI_sub',
                             'FamousShape', 'FamousShape3k', 'FamousShape5k', 'FamousShape50k', 'Others', 'WireframePC', 'NestPC'])
parser.add_argument('--ckpt_dirs', type=str, default='000')
parser.add_argument('--ckpt_iters', type=str, default='800')
args = parser.parse_args()

encode_knn = 16
train_patch_size = 800
train_batch_size = 75

if args.mode == 'train':
    trainset_list = 'trainingset_whitenoise'

    os.system('CUDA_VISIBLE_DEVICES={} python train.py --dataset_root={} --log_root={} --trainset_list={} --patch_size={} --batch_size={} \
                                                       --encode_knn={}'.format(
        args.gpu, args.dataset_root, args.log_root, trainset_list, train_patch_size, train_batch_size, encode_knn))

elif args.mode == 'test':
    test_patch_size = train_patch_size
    test_batch_size = 550

    if args.ckpt_dirs == '':
        args.ckpt_dirs = os.path.split(os.path.abspath(os.path.dirname(os.getcwd())))[-1]

    save_pn = False          # to save the point normals (.npy) or not
    sparse_patch = True      # to output sparse point normals or not

    testset_list = f'testset_{args.data_set}'
    eval_list = testset_list

    if args.data_set == 'PCPNet':
        eval_list = 'testset_no_noise testset_low_noise testset_med_noise testset_high_noise ' \
                    + 'testset_vardensity_striped testset_vardensity_gradient'
    elif args.data_set == 'FamousShape':
        eval_list = 'testset_noise_clean testset_noise_low testset_noise_med testset_noise_high ' \
                    + 'testset_density_stripe testset_density_gradient'
    elif args.data_set == 'SceneNN':
        eval_list = 'testset_SceneNN_clean testset_SceneNN_noise'
    elif args.data_set == 'WireframePC':
        test_patch_size = 100

    if args.data_set in ['ScanNet', 'Semantic3D', 'KITTI_sub', 'Others', 'NestPC', 'WireframePC']:
        save_pn = True
        sparse_patch = False

    command = f'CUDA_VISIBLE_DEVICES={args.gpu} python test.py --dataset_root={args.dataset_root} --data_set={args.data_set} ' \
            + f'--log_root={args.log_root} --ckpt_dirs={args.ckpt_dirs} --ckpt_iters={args.ckpt_iters} ' \
            + f'--patch_size={test_patch_size} --batch_size={test_batch_size} --encode_knn={encode_knn} ' \
            + f'--save_pn={save_pn} --sparse_patch={sparse_patch} --testset_list={testset_list} --eval_list {eval_list} '
    print(command)
    os.system(command)

else:
    print('The mode is unsupported!')