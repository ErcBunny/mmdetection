_base_ = './cascade-mask-rcnn_vitmae_b16_sfpn_2_768_1x_coco.py'
model = dict(
    backbone=dict(
        weight_url=""
    )
)