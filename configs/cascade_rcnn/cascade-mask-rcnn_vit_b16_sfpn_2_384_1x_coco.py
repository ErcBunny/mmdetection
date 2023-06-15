_base_ = './cascade-mask-rcnn_vitmae_b16_sfpn_2_384_1x_coco.py'
model = dict(
    backbone=dict(
        weight_url=""
    )
)