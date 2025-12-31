# OSS3D
Code for OSS3D: Object-Guided Semi-Supervised Bird's-Eye View 3D Object Detection with 3D Box Refinement.

## 环境准备

### 1) 创建并激活 Conda 环境

```bash
conda create -n oss3d python=3.8 -y
conda activate oss3d
```

### 2) 安装本项目（editable）

在仓库根目录执行：

```bash
pip install -e .
```

## 数据准备

使用 `tools/create_data_bevdet.py` 进行 nuScenes 数据相关的 pkl 生成/增强/半监督划分（子命令：`prep` / `add_ann` / `semi`）。

### 0) 下载 nuScenes 数据 & 建立本地软链接

- **下载数据**：到官方页面按指引下载（建议先下载 `Full dataset (v1.0)`，并根据需要下载 `Trainval` / `Mini` 等 split）。
  - 官方下载入口：[`https://www.nuscenes.org/download`](https://www.nuscenes.org/download)

- **建立软链接到本项目**：假设你已经把 nuScenes 数据解压到本机某个目录（例如 `/data/nuscenes`），将其软链接到仓库的 `./data/nuscenes`：

```bash
mkdir -p ./data
ln -s /data/nuscenes ./data/nuscenes
```

> 如果你本机路径不同，把 `/data/nuscenes` 替换成你的实际数据目录即可。

### 1) 生成 infos（prep）

**输出位置与命名（默认）**：在 `--root-path` 目录下生成：

- `{extra_tag}_infos_train.pkl`
- `{extra_tag}_infos_val.pkl`

例如默认参数会生成：

- `./data/nuscenes/nuscenes_infos_train.pkl`
- `./data/nuscenes/nuscenes_infos_val.pkl`

```bash
python tools/create_data_bevdet.py prep \
  --root-path ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-trainval \
  --max-sweeps 0
```

### 2) 给 infos 增加 ann/gt/occ_path（add_ann）

**读取与写回（默认）**：会读取并覆盖写回同名文件（在 `./data/nuscenes/` 下）：

- `./data/nuscenes/{extra_tag}_infos_train.pkl`
- `./data/nuscenes/{extra_tag}_infos_val.pkl`

```bash
python tools/create_data_bevdet.py add_ann \
  --extra-tag nuscenes \
  --nuscenes-version v1.0-trainval \
  --dataroot ./data/nuscenes/
```

### 3) 生成半监督 labeled/unlabeled infos（semi）

原脚本逻辑：当 `id % ratio == 0` 时划为 labeled，其余为 unlabeled；输出文件名中会用到 `percent`（unlabeled 为 `100 - percent`）。

**输出位置与命名（默认）**：在 `--root-path` 目录下生成：

- labeled：`{extra_tag}_infos_labeled_{percent}.pkl`
- unlabeled：`{extra_tag}_infos_unlabeled_{100 - percent}.pkl`

例如 `--extra-tag nuscenes --percent 10` 会生成：

- `./data/nuscenes/nuscenes_infos_labeled_10.pkl`
- `./data/nuscenes/nuscenes_infos_unlabeled_90.pkl`

```bash
# 生成 10% 有监督数据（ratio=10 -> percent=10）
python tools/create_data_bevdet.py semi \
  --extra-tag nuscenes \
  --root-path ./data/nuscenes \
  --ratio 10 \
  --percent 10 \
  --version v1.0-trainval

# 生成 20% 有监督数据（ratio=5 -> percent=20）
python tools/create_data_bevdet.py semi \
  --extra-tag nuscenes \
  --root-path ./data/nuscenes \
  --ratio 5 \
  --percent 20 \
  --version v1.0-trainval

# 生成 30% 有监督数据（ratio=3 -> percent=30）
python tools/create_data_bevdet.py semi \
  --extra-tag nuscenes \
  --root-path ./data/nuscenes \
  --ratio 3 \
  --percent 30 \
  --version v1.0-trainval
```

## 训练

### 1) 有监督预训练（以 10% / 20% / 30% 为例）

在开始训练前，请确保已经生成了对应的 pkl（见上面的“数据准备”）：

- `data/nuscenes/nuscenes_infos_val.pkl`
- `data/nuscenes/nuscenes_infos_labeled_{10,20,30}.pkl`

使用下面的配置进行纯有监督训练（建议指定 `--work-dir`，方便后续半监督阶段直接引用 checkpoint）：

```bash
# 10% supervised pretrain
python tools/train.py configs/bevdet/bevdet-r50-10per.py --work-dir results/bevdet-r50-10per

# 20% supervised pretrain
python tools/train.py configs/bevdet/bevdet-r50-20per.py --work-dir results/bevdet-r50-20per

# 30% supervised pretrain
python tools/train.py configs/bevdet/bevdet-r50-30per.py --work-dir results/bevdet-r50-30per
```

BEVDepth 的有监督预训练运行方式与 BEVDet **一致**，仅配置文件不同：

```bash
# 10% supervised pretrain (BEVDepth)
python tools/train.py configs/bevdepth/bevdepth-r50-10per.py --work-dir results/bevdepth-r50-10per

# 20% supervised pretrain (BEVDepth)
python tools/train.py configs/bevdepth/bevdepth-r50-20per.py --work-dir results/bevdepth-r50-20per

# 30% supervised pretrain (BEVDepth)
python tools/train.py configs/bevdepth/bevdepth-r50-30per.py --work-dir results/bevdepth-r50-30per
```

训练产物默认在 `--work-dir` 下（例如 `latest.pth` / `epoch_*.pth`）。

### 2) 半监督训练（以 30% 为例）

半监督阶段需要使用 30% 对应的 labeled/unlabeled pkl（见“数据准备”），并在训练时 **load-from** 第 1 阶段对应模型的有监督预训练权重。

- **BEVDet (30%)**：
  - 训练配置：`configs/semi_bevdet/bevdet-r50-30per-depth.py`
  - load-from：`results/bevdet-r50-30per/latest.pth`（或你想加载的 `epoch_*.pth`）

```bash
python tools/train.py configs/semi_bevdet/bevdet-r50-30per-depth.py \
  --work-dir results/semi_bevdet-r50-30per-depth \
  --load-from results/bevdet-r50-30per/latest.pth
```

- **BEVDepth (30%)**：
  - 训练配置：`configs/semi_bevdepth/bevdepth-r50-30per-depth.py`
  - load-from：`results/bevdepth-r50-30per/latest.pth`（或你想加载的 `epoch_*.pth`）

```bash
python tools/train.py configs/semi_bevdepth/bevdepth-r50-30per-depth.py \
  --work-dir results/semi_bevdepth-r50-30per-depth \
  --load-from results/bevdepth-r50-30per/latest.pth
```

### 3) 最终测试 / 评估（test_semi.py）

使用 `tools/test_semi.py` 对训练得到的 checkpoint 进行测试与评估（支持单卡与分布式）。

#### 单卡测试

```bash
# 示例：评估 bbox（按需替换 CONFIG / CHECKPOINT）
python tools/test_semi.py CONFIG.py CHECKPOINT.pth --eval bbox
```

#### 分布式测试

仓库提供了脚本 `tools/dist_test.sh`：

```bash
# 用法：bash tools/dist_test.sh CONFIG CHECKPOINT GPUS [--eval bbox ...]
bash tools/dist_test.sh CONFIG.py CHECKPOINT.pth 8 --eval bbox
```

#### 半监督模型推理分支（student / teacher）

`test_semi.py` 会读取配置里的 `cfg.model.test_cfg.inference_on`（默认 `student`），并对对应分支进行推理：

- `student`：使用学生模型推理
- `teacher`：使用教师模型推理（如果你的模型/配置支持）

你可以在测试时用 `--cfg-options` 覆盖，例如：

```bash
python tools/test_semi.py CONFIG.py CHECKPOINT.pth \
  --eval bbox \
  --cfg-options model.test_cfg.inference_on=teacher
```
