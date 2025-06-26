from mmcv import Config
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmseg.models.backbones.femtonet
import mmseg.datasets.pipelines.align_resize

# 配置文件路径和模型权重路径
config_file = 'configs/femtonet/femtonet.py'
checkpoint_file = 'log/fempvit_m1_1_human_stage2/latest.pth'
img_path = 'tools/0_00420.png'  # 待预测图像路径
device = 'cuda:0'  # 如果用CPU则改为 'cpu'
save_path = 'tools/0_00420_seg.png'
# =======================================

# 加载 config
cfg = Config.fromfile(config_file)

# 构建模型并载入 checkpoint
model = init_segmentor(cfg, checkpoint_file, device=device)

# 推理图像
result = inference_segmentor(model, img_path)

# 可视化并保存结果（含透明 mask）
# palette：模型在训练集中的类别颜色，默认为 `dataset.PALETTE`
show_result_pyplot(model, img_path, result, out_file=save_path)
