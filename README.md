
(ActionRecognition) user@user-System-Product-Name:~/projects/mmaction2-main$ python tools/train.py configs/detection/slowfast/slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py
/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/optim/optimizer/zero_optimizer.py:11: DeprecationWarning: `TorchScript` support for functional optimizers is deprecated and will be removed in a future PyTorch release. Consider using the `torch.compile` optimizer instead.
  from torch.distributed.optim import \
01/05 10:58:31 - mmengine - INFO -
------------------------------------------------------------
System environment:
    sys.platform: linux
    Python: 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0]
    CUDA available: True
    MUSA available: False
    numpy_random_seed: 942388073
    GPU 0,1,2,3: NVIDIA H100 80GB HBM3
    CUDA_HOME: /usr/local/cuda-12.1
    NVCC: Cuda compilation tools, release 12.1, V12.1.66
    GCC: gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
    PyTorch: 2.4.1+cu121
    PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201703
  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v3.4.2 (Git Hash 1137e04ec0b5251ca2b4400a4fd3c667ce843d67)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX512
  - CUDA Runtime 12.1
  - NVCC architecture flags: -gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90
  - CuDNN 90.1  (built against CUDA 12.4)
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=12.1, CUDNN_VERSION=9.1.0, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=pedantic -Wno-error=old-style-cast -Wno-missing-braces -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=2.4.1, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,

    TorchVision: 0.19.1+cu121
    OpenCV: 4.12.0
    MMEngine: 0.10.7

Runtime environment:
    cudnn_benchmark: False
    mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
    dist_cfg: {'backend': 'nccl'}
    seed: 942388073
    diff_rank_seed: False
    deterministic: False
    Distributed launcher: none
    Distributed training: False
    GPU number: 1
------------------------------------------------------------

01/05 10:58:32 - mmengine - INFO - Config:
ann_file_train = '/home/user/datasets/ava/annotations/ava_train_v2.1.csv'
ann_file_val = '/home/user/datasets/ava/annotations/ava_val_v2.1.csv'
anno_root = '/home/user/datasets/ava/annotations'
auto_scale_lr = dict(base_batch_size=128, enable=False)
data_root = '/home/user/datasets/ava/rawframes'
dataset_type = 'AVADataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
exclude_file_train = '/home/user/datasets/ava/annotations/ava_train_excluded_timestamps_v2.1.csv'
exclude_file_val = '/home/user/datasets/ava/annotations/ava_val_excluded_timestamps_v2.1.csv'
file_client_args = dict(io_backend='disk')
label_file = '/home/user/datasets/ava/annotations/ava_action_list_v2.1.pbtxt'
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    _scope_='mmdet',
    backbone=dict(
        channel_ratio=8,
        fast_pathway=dict(
            base_channels=8,
            conv1_kernel=(
                5,
                7,
                7,
            ),
            conv1_stride_t=1,
            depth=50,
            lateral=False,
            pool1_stride_t=1,
            pretrained=None,
            spatial_strides=(
                1,
                2,
                2,
                1,
            ),
            type='resnet3d'),
        pretrained=None,
        resample_rate=8,
        slow_pathway=dict(
            conv1_kernel=(
                1,
                7,
                7,
            ),
            conv1_stride_t=1,
            depth=50,
            dilations=(
                1,
                1,
                1,
                1,
            ),
            inflate=(
                0,
                0,
                1,
                1,
            ),
            lateral=True,
            pool1_stride_t=1,
            pretrained=None,
            spatial_strides=(
                1,
                2,
                2,
                1,
            ),
            type='resnet3d'),
        speed_ratio=8,
        type='mmaction.ResNet3dSlowFast'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmaction.ActionDataPreprocessor'),
    init_cfg=dict(
        checkpoint=
        'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth',
        type='Pretrained'),
    roi_head=dict(
        bbox_head=dict(
            background_class=True,
            dropout_ratio=0.5,
            in_channels=2304,
            multilabel=True,
            num_classes=9,
            type='BBoxHeadAVA'),
        bbox_roi_extractor=dict(
            output_size=8,
            roi_layer_type='RoIAlign',
            type='SingleRoIExtractor3D',
            with_temporal_pool=True),
        type='AVARoIHead'),
    test_cfg=dict(rcnn=None),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                min_pos_iou=0.9,
                neg_iou_thr=0.9,
                pos_iou_thr=0.9,
                type='MaxIoUAssignerAVA'),
            pos_weight=1.0,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=32,
                pos_fraction=1,
                type='RandomSampler'))),
    type='FastRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.2, momentum=0.9, type='SGD', weight_decay=1e-05))
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        gamma=0.1,
        milestones=[
            10,
            15,
        ],
        type='MultiStepLR'),
]
proposal_file_train = '/home/user/datasets/ava/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl'
proposal_file_val = '/home/user/datasets/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl'
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/home/user/datasets/ava/annotations/ava_val_v2.1.csv',
        data_prefix=dict(img='/home/user/datasets/ava/rawframes'),
        exclude_file=
        '/home/user/datasets/ava/annotations/ava_val_excluded_timestamps_v2.1.csv',
        label_file=
        '/home/user/datasets/ava/annotations/ava_action_list_v2.1.pbtxt',
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                test_mode=True,
                type='SampleAVAFrames'),
            dict(io_backend='disk', type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        '/home/user/datasets/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        test_mode=True,
        type='AVADataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home/user/datasets/ava/annotations/ava_val_v2.1.csv',
    exclude_file=
    '/home/user/datasets/ava/annotations/ava_val_excluded_timestamps_v2.1.csv',
    label_file='/home/user/datasets/ava/annotations/ava_action_list_v2.1.pbtxt',
    type='AVAMetric')
train_cfg = dict(
    max_epochs=20, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='/home/user/datasets/ava/annotations/ava_train_v2.1.csv',
        data_prefix=dict(img='/home/user/datasets/ava/rawframes'),
        exclude_file=
        '/home/user/datasets/ava/annotations/ava_train_excluded_timestamps_v2.1.csv',
        label_file=
        '/home/user/datasets/ava/annotations/ava_action_list_v2.1.pbtxt',
        pipeline=[
            dict(clip_len=32, frame_interval=2, type='SampleAVAFrames'),
            dict(io_backend='disk', type='RawFrameDecode'),
            dict(scale_range=(
                256,
                320,
            ), type='RandomRescale'),
            dict(size=256, type='RandomCrop'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        '/home/user/datasets/ava/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl',
        type='AVADataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=32, frame_interval=2, type='SampleAVAFrames'),
    dict(io_backend='disk', type='RawFrameDecode'),
    dict(scale_range=(
        256,
        320,
    ), type='RandomRescale'),
    dict(size=256, type='RandomCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(collapse=True, input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
url = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth'
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/home/user/datasets/ava/annotations/ava_val_v2.1.csv',
        data_prefix=dict(img='/home/user/datasets/ava/rawframes'),
        exclude_file=
        '/home/user/datasets/ava/annotations/ava_val_excluded_timestamps_v2.1.csv',
        label_file=
        '/home/user/datasets/ava/annotations/ava_action_list_v2.1.pbtxt',
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=2,
                test_mode=True,
                type='SampleAVAFrames'),
            dict(io_backend='disk', type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        '/home/user/datasets/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        test_mode=True,
        type='AVADataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/home/user/datasets/ava/annotations/ava_val_v2.1.csv',
    exclude_file=
    '/home/user/datasets/ava/annotations/ava_val_excluded_timestamps_v2.1.csv',
    label_file='/home/user/datasets/ava/annotations/ava_action_list_v2.1.pbtxt',
    type='AVAMetric')
val_pipeline = [
    dict(
        clip_len=32, frame_interval=2, test_mode=True, type='SampleAVAFrames'),
    dict(io_backend='disk', type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(collapse=True, input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb'

01/05 10:58:35 - mmengine - INFO - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
01/05 10:58:35 - mmengine - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) RuntimeInfoHook
(BELOW_NORMAL) LoggerHook
 --------------------
before_train:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(VERY_LOW    ) CheckpointHook
 --------------------
before_train_epoch:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(NORMAL      ) DistSamplerSeedHook
 --------------------
before_train_iter:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
 --------------------
after_train_iter:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
(LOW         ) ParamSchedulerHook
(VERY_LOW    ) CheckpointHook
 --------------------
after_train_epoch:
(NORMAL      ) IterTimerHook
(NORMAL      ) SyncBuffersHook
(LOW         ) ParamSchedulerHook
(VERY_LOW    ) CheckpointHook
 --------------------
before_val:
(VERY_HIGH   ) RuntimeInfoHook
 --------------------
before_val_epoch:
(NORMAL      ) IterTimerHook
(NORMAL      ) SyncBuffersHook
 --------------------
before_val_iter:
(NORMAL      ) IterTimerHook
 --------------------
after_val_iter:
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
 --------------------
after_val_epoch:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
(LOW         ) ParamSchedulerHook
(VERY_LOW    ) CheckpointHook
 --------------------
after_val:
(VERY_HIGH   ) RuntimeInfoHook
 --------------------
after_train:
(VERY_HIGH   ) RuntimeInfoHook
(VERY_LOW    ) CheckpointHook
 --------------------
before_test:
(VERY_HIGH   ) RuntimeInfoHook
 --------------------
before_test_epoch:
(NORMAL      ) IterTimerHook
 --------------------
before_test_iter:
(NORMAL      ) IterTimerHook
 --------------------
after_test_iter:
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
 --------------------
after_test_epoch:
(VERY_HIGH   ) RuntimeInfoHook
(NORMAL      ) IterTimerHook
(BELOW_NORMAL) LoggerHook
 --------------------
after_test:
(VERY_HIGH   ) RuntimeInfoHook
 --------------------
after_run:
(BELOW_NORMAL) LoggerHook
 --------------------
01/05 10:58:37 - mmengine - INFO - 56 out of 56 frames are valid.
01/05 10:58:37 - mmengine - INFO - 8 out of 8 frames are valid.
01/05 10:58:39 - mmengine - INFO - load model from: https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth
01/05 10:58:39 - mmengine - INFO - Loads checkpoint by http backend from path: https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth
01/05 10:58:39 - mmengine - WARNING - The model and loaded state dict do not match exactly

unexpected key in source state_dict: cls_head.fc_cls.weight, cls_head.fc_cls.bias

missing keys in source state_dict: roi_head.bbox_head.fc_cls.weight, roi_head.bbox_head.fc_cls.bias

01/05 10:58:39 - mmengine - WARNING - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
01/05 10:58:39 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
01/05 10:58:39 - mmengine - INFO - Checkpoints will be saved to /home/user/projects/mmaction2-main/work_dirs/slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.
Traceback (most recent call last):
  File "/home/user/projects/mmaction2-main/tools/train.py", line 143, in <module>
    main()
  File "/home/user/projects/mmaction2-main/tools/train.py", line 139, in main
    runner.train()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/runner/runner.py", line 1777, in train
    model = self.train_loop.run()  # type: ignore
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/runner/loops.py", line 98, in run
    self.run_epoch()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/runner/loops.py", line 114, in run_epoch
    for idx, data_batch in enumerate(self.dataloader):
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1344, in _next_data
    return self._process_data(data)
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1370, in _process_data
    data.reraise()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/_utils.py", line 706, in reraise
    raise exception
FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 309, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 410, in __getitem__
    data = self.prepare_data(idx)
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 793, in prepare_data
    return self.pipeline(data_info)
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 60, in __call__
    data = t(data)
  File "/home/user/thirdkt/mmcv-2.1.0/mmcv/transforms/base.py", line 12, in __call__
    return self.transform(results)
  File "/home/user/projects/mmaction2-main/mmaction/datasets/transforms/loading.py", line 1429, in transform
    img_bytes = self.file_client.get(filepath)
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/fileio/file_client.py", line 301, in get
    return self.client.get(filepath)
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/fileio/backends/local_backend.py", line 33, in get
    with open(filepath, 'rb') as f:
FileNotFoundError: [Errno 2] No such file or directory: '/home/user/datasets/ava/rawframes/1/img_05399.jpg'
