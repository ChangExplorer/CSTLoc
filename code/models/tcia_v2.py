# TCIA-v2：继承3.6结构，禁止复用旧GT-crop权重；本文件仅作版本隔离。
from models.tcia import TCIA as _TCIA


class TCIAv2(_TCIA):
    MODEL_VERSION = 'tcia_v2_peakcrop_rank4_attn256'
    pass
