# E2E RL Finetuning Configuration
# Based on R2SE e2e_r2se_rlft.py with E2E_RL enhancements

_base_ = ["../../_base_/datasets/nus-3d.py",
          "../../_base_/default_runtime.py"]

# ============ Basic Settings ============
plugin = True
plugin_dir = "mmdet3d_plugin/"
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# nuPlan/openScenes/NavSim classes
class_names = ['vehicle', 'bicycle', 'pedestrian',
               'traffic_cone', 'barrier', 'czone_sign', 'generic_object']
vehicle_id_list = [0, 1]
group_id_list = [[0], [1], [2], [3, 4, 5, 6]]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)

# ============ Model Dimensions ============
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
queue_length = 4

# ============ Planning Args ============
predict_steps = 8
predict_modes = 6
use_nonlinear_optimizer = True
planning_steps = 8
use_col_optim = False

# ============ Tracking Args ============
past_steps = 3
fut_steps = 4

# ============ OCC Args ============
occ_past = 3
occ_future = 8
occflow_grid_conf = {
    'xbound': [-51.2, 51.2, 0.512],
    'ybound': [-51.2, 51.2, 0.512],
    'zbound': [-10.0, 10.0, 20.0],
}

# ============ Data Path ============
train_dataset_type = "NavSimOpenScenesE2EFineTune"
dataset_type = "NavSimOpenScenesE2EFineTune"
file_client_args = dict(backend="disk")

data_root = "./data/openscene-v1.1/"
info_root = data_root + "paradrive_infos_v2/"
img_root_train = data_root + "sensor_blobs/trainval"
img_root_test = data_root + "sensor_blobs/test"

ann_file_train = info_root + "nuplan_navsim_train.pkl"
ann_file_val = info_root + "nuplan_navsim_test.pkl"
ann_file_test = info_root + "nuplan_navsim_test.pkl"
nav_filter_path_train = "path_to_yaml/navtrain.yaml"
nav_filter_path_val = "path_to_yaml/navtest.yaml"
nav_filter_path_test = "path_to_yaml/navtest.yaml"

finetune_yaml = [
    "data_loop/navtrain_split/e2e_hydramdp_ep8/navtrain_collision.yaml",
    "data_loop/navtrain_split/e2e_hydramdp_ep8/navtrain_ep_1pct.yaml",
    "data_loop/navtrain_split/e2e_hydramdp_ep8/navtrain_off_road.yaml",
]

num_task = 12

# ============ Model Config ============
model = dict(
    type="BEVHydra",
    gt_iou_threshold=train_gt_iou_threshold,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=len(class_names),
    vehicle_id_list=vehicle_id_list,
    pc_range=point_cloud_range,
    
    # LoRA Finetuning
    lora_finetuning=True,
    
    # ===== Backbone (Frozen) =====
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        norm_eval=False,
        style='caffe',
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_bn=True,
    freeze_bev_encoder=True,
    
    # ===== Detection Head (Frozen) =====
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),
    mem_args=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=len(class_names),
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
    ),
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        # ... (same as R2SE)
    ),
    
    # ===== E2E RL Finetuning Head =====
    planning_head=dict(
        type='HydraTrajEnsHeadE2E',  # E2E RL enhanced head
        use_lora=True,
        trans_use_lora=True,
        rl_finetuning=True,
        importance_sampling=True,
        moe_lora=True,
        num_task=num_task,
        
        # E2E RL specific settings
        enable_e2e_rl=True,  # NEW: Enable E2E RL finetuning
        closed_loop_reward_config=dict(
            collision_weight=0.5,
            offroad_weight=0.5,
            progress_weight=8.0,
            comfort_weight=2.0,
            ttc_weight=5.0,
        ),
        
        # Conservative RL settings
        conservative_rl_config=dict(
            kl_target=0.01,
            kl_penalty_weight=1.0,
            use_reference_anchor=True,
            reference_anchor_weight=0.1,
            use_stapo_filter=True,
            stapo_threshold=0.0,
            max_step_size=0.1,
            ppo_eps=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            value_loss_weight=0.5,
        ),
        
        # Standard head settings
        num_poses=40,
        d_ffn=256 * 4,
        d_model=256,
        vocab_path='test_8192_kmeans.npy',
        nhead=8,
        nlayers=1,
        num_commands=4,
        transformer_decoder=dict(
            type='BEVOnlyMotionTransformerDecoder',
            pc_range=point_cloud_range,
            embed_dims=_dim_,
            num_layers=3,
            transformerlayers=dict(
                type='MotionTransformerAttentionLayer',
                batch_first=True,
                use_lora=True,
                lora_rank=16,
                moe_lora=True,
                num_task=num_task,
                attn_cfgs=[
                    dict(
                        type='MotionDeformableAttention',
                        num_steps=predict_steps,
                        embed_dims=_dim_,
                        num_levels=1,
                        num_heads=8,
                        num_points=4,
                        sample_index=-1,
                        use_lora=True,
                        moe_lora=True,
                        num_task=num_task,
                        lora_rank=16
                    ),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
        ),
        bev_h=bev_h_,
        bev_w=bev_w_,
    ),
)

# ============ E2E RL Training Pipeline ============
e2e_rl_train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph",
        to_float32=True,
        file_client_args=file_client_args,
        img_root=img_root_train
        ),
    dict(type="ScaleMultiViewImage3D", scale=0.5),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=False,
        with_ins_inds_3d=True,
        ins_inds_add_1=True,
    ),
    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "img",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "sdc_planning",
            "sdc_planning_mask",
            "command",
            "sdc_planning_world",
            "sdc_planning_past",
            "sdc_planning_mask_past",
            "gt_pre_command_sdc",
            "sdc_status",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "score",
            "fail_mask",
            # NEW: E2E RL specific data
            "gt_trajectory",  # For closed-loop reward
            "map_features",   # For off-road checking
            "agent_trajectories",  # For collision checking
        ],
    ),
]

# ============ Data Config ============
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=train_dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        nav_filter_path=nav_filter_path_train,
        pipeline=e2e_rl_train_pipeline,  # Use E2E RL pipeline
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        planning_steps=planning_steps,
        occ_receptive_field=occ_past+1,
        occ_n_future=occ_future,
        occ_filter_invalid_sample=False,
        load_interval=1,
        box_type_3d="LiDAR",
        fix_can_bus_rotation=True,
        finetune_yaml=finetune_yaml,
        
        # E2E RL settings
        use_e2e_rl=True,
        enable_closed_loop_reward=True,
    ),
    val=dict(
        type=dataset_type,
        val_fail_anno_info=ann_file_train,
        train_img_root=img_root_train,
        finetune_yaml=finetune_yaml,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        use_valid_flag=True,
        ann_file=ann_file_val,
        nav_filter_path=nav_filter_path_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=[],
        planning_steps=planning_steps,
        occ_receptive_field=occ_past+1,
        occ_n_future=occ_future,
        occ_filter_invalid_sample=False,
        fix_can_bus_rotation=True,
    ),
    test=dict(
        type=dataset_type,
        use_num_split=5,
        val_fail_anno_info=ann_file_train,
        train_img_root=img_root_train,
        finetune_yaml=finetune_yaml,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        use_valid_flag=True,
        ann_file=ann_file_test,
        nav_filter_path=nav_filter_path_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        occ_n_future=occ_future,
        planning_steps=planning_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=[],
        fix_can_bus_rotation=True,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

# ============ Optimizer ============
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            # E2E RL specific: Lower learning rate for LoRA
            "planning_head": dict(lr_mult=0.5),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# ============ Learning Schedule ============
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 7
evaluation = dict(interval=7, pipeline=test_pipeline)
runner = dict(type="EpochBasedRunnerAutoResume", max_epochs=total_epochs)

log_config = dict(
    interval=10, hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
        # E2E RL specific logging
        dict(type="E2ERLMetricsLogger"),  # NEW: Log RL metrics
    ]
)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
load_from = "ckpts/e2e_100pct_exps/e2e_hydramdp_ep8.pth"
find_unused_parameters = True

# ============ E2E RL Specific Settings ============
e2e_rl_settings = dict(
    # Reward computation
    use_closed_loop_reward=True,
    reward_normalization=True,
    
    # Conservative updates
    enable_kl_constraint=True,
    enable_reference_anchor=True,
    enable_stapo_filter=True,
    
    # Training stability
    gradient_accumulation_steps=4,
    use_mixed_precision=True,
    ema_decay=0.999,
    
    # Reference model update
    reference_update_freq=100,
    reference_tau=0.01,
)
