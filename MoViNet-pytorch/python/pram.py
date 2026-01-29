from movinets import MoViNet
from movinets.config import _C

model = MoViNet(_C.MODEL.MoViNetA2, causal=True, pretrained=True)

params = sum(p.numel() for p in model.parameters())
print(params)
