# _base_ = "bevdepth-r50_default_dt.py"
_base_ = "bevdepth-r50_default_rand_aug.py"

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'


data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_labeled_10.pkl',
        ),
        unsup=dict(
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_unlabeled_90.pkl',
        )
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[2,1]
        )
    )
)

custom_hooks = [
    #dict(type="WeightSummary"),
    dict(
        type="MeanTeacher",
        momentum=0.999, 
        interval=1, 
        start_steps=0,
        skip_buffer=False
    )
]

model = dict(
    type="semi_bevdet_gs_v4_depth", 
    train_cfg=dict(
        iter=0,
        burn_in_steps=0,
        sup_weight=1.0,
        unsup_weight=2.0,
        use_history_box_fusion=True,
        history_capacity=10,
        adaptive_threshold=True,
    )
)
# load_from = 'results/bevdepth-r50-30per/epoch_24_ema.pth'