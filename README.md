# rea-video

参考片 → 换成你自己的商品 → 一条成片。程序自动完成拆镜、动作改写、H3 重生成、配音改写、验收；人工只做逐镜复核。

本仓库是 `new-hypit-rea-video` 这台机器上跑通的完整实现，可直接在另一台装有 NVIDIA GPU 的机器上部署。上游分析/编排逻辑来自 [JuneYaooo/rea-video](https://github.com/JuneYaooo/rea-video)（commit `a124acd4a02b3a9b72b9e7dedca5d1bfcda2a7b8`），本仓库补上本地 H3 执行、商品契约、验收门、配音改写、任务队列和网页服务。

---

## 1. 它到底做什么

输入：

- 一条参考视频（带原声/口播）
- 1–3 张你自己的商品图

输出：

- 一条与参考片同结构、同节奏的成片，商品换成你的，口播改成符合你商品的文案，原片 BGM/环境声保留，H3 生成的分镜音轨作为底噪保留
- `remake/acceptance.json` 验收报告、逐镜复核记录、可选网页队列

硬约束（写在程序里，不靠主 agent 事后改）：

1. **商品身份**：每镜提示词都带商品事实，片尾不再出现原片商品
2. **动作兼容**：原片在系鞋带、拆鞋盒，而你的商品没有鞋带/鞋盒 → 程序自动改动作，不脑补不存在的部件
3. **口播改写**：只允许说商品事实里有的颜色、部件、卖点；ASR 回读校验，低于 0.85 拒绝
4. **验收门**：结构、口播、逐镜复核任一不过 → 不出片，进程非 0
5. **队列**：排队、取消、重排、断点续跑（必须人工确认对账结果）

---

## 2. 总流程

```
参考视频 ──analyze──> breakdown.json（镜头、原口播、原声）
     │
商品图 ──plan──────> actions.json（每镜动作/是否改写）
     │                 contract.json（商品有哪些部件、缺哪些）
     │                 prompts.json / graphs/G*.json（8 个 H3 调用）
     │
     └──render────> H3 逐段生成（带音频）→ 精确拼接 → 配音改写 → 混音 → 验收门
```

单个 H3 调用固定 124 帧 / 24 fps / 480×864 / 4 步 turbo，最终 30 fps 1080×1920 成片。

---

## 3. 环境要求

| 项 | 实测值 | 说明 |
| --- | --- | --- |
| OS | Windows 11 + Git Bash | 启动脚本用了 `cygpath`，Linux 需改成普通路径 |
| GPU | RTX 3090 24 GB | 实测峰值 21.9 GB；24 GB 以下不建议 |
| ComfyUI | 0.44.0 | `G:/comfyui` |
| Python | 3.12.10 | 直接用 ComfyUI 的 venv `G:/comfyui/venv` |
| PyTorch | 2.9.1+cu130 | ComfyUI venv 自带 2.8.0+cu126 不够；H3 用 cu130 内核 |
| ffmpeg / ffprobe | PATH 内 | 拼接、混音、抽帧 |
| uv | 可选 | 用 `uv run --with` 隔离依赖时用；也可 `pip install -r requirements.txt` |
| llama.cpp | 任意带 `llama-server` 的 CUDA12 构建 | 规划阶段看图用，不参与出片 |

---

## 4. 部署步骤

### 4.1 ComfyUI 与插件

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git /g/comfyui
cd /g/comfyui
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt

cd custom_nodes
git clone https://github.com/MiniMax-AI/ComfyUI_MiniMaxH3_Director.git
git -C ComfyUI_MiniMaxH3_Director checkout 4c6245e518dd8a494f3afc4f3cccd53f22d82d10
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git -C ComfyUI-VideoHelperSuite checkout 2c75dc1b379a59e0880619d029607af5169a62ef
git clone https://github.com/kijai/ComfyUI-KJNodes.git
git -C ComfyUI-KJNodes checkout ab72eda7ffad6280422d71ab7d54d3fb87c0ba18
```

三个插件版本必须对齐上表 commit；H3 的节点输入 schema 会随插件变化，换版本可能直接构图失败。

### 4.1.1 PyTorch cu130

H3 图需要 CUDA 13 内核，普通的 ComfyUI venv（cu126）会在采样阶段失败。两种做法：

```bash
# A. 直接装进 ComfyUI venv
/g/comfyui/venv/Scripts/python.exe -m pip install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# B. 单独装到 workspace 目录，不污染 venv
/g/comfyui/venv/Scripts/python.exe -m pip install --target engine/torch-cu130 \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
export REA_TORCH_DIR=/path/to/rea-video/engine/torch-cu130
```

`engine/launch_engine.py` 只在 `REA_TORCH_DIR`（或 `engine/torch-cu130`）存在时把它插到 `sys.path` 最前；不存在就直接用 venv 里的 torch。

### 4.2 模型

全部来自官方 repackage 仓库 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)，文件名与目录必须完全一致：

| 放到 | 文件 | 大小 |
| --- | --- | --- |
| `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20.97 GB |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.69 GB |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.14 GB（模板里的默认值） |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB |
| `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB |
| `models/loras/` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.96 GB |

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors \
  --local-dir /g/comfyui/models
```

出片时 `engine/rea_remake.py` 会把节点 2 的编码器强制换成 AWQ 版本（更快、更省显存）；想换回 int8，设 `REA_CLIP_NAME=qwen3vl_32b_minimax_h3_int8_convrot.safetensors`。模板 `armA_seg01_graph.json` 里写的 int8 只是占位。

### 4.3 规划用视觉模型（可选但 analyze/plan 必需）

```bash
hf download ggml-org/Qwen2.5-Omni-7B-GGUF \
  Qwen2.5-Omni-7B-Q4_K_M.gguf mmproj-Qwen2.5-Omni-7B-Q8_0.gguf \
  --local-dir /g/comfyui/models/LLM/Qwen2.5-Omni-7B-GGUF
```

`llama-server` 默认按 LM Studio 的目录布局找：

```
~/.lmstudio/extensions/backends/llama.cpp-*/llama-server.exe
~/.lmstudio/extensions/backends/vendor/win-llama-cuda12-vendor-*/
```

不在这个布局时用两个环境变量指过去：

```bash
export REA_VISION_EXE=/path/to/llama-server.exe
export REA_VISION_VENDOR=/path/to/vendor  # 放 cublas64_12.dll 的目录
export REA_VISION_MODELS=/g/comfyui/models/LLM/Qwen2.5-Omni-7B-GGUF
```

### 4.4 ASR 模型（配音回读校验）

```bash
hf download csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
  --local-dir engine/rea-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
```

### 4.5 Python 依赖

```bash
cd /path/to/rea-video
/g/comfyui/venv/Scripts/python.exe -m pip install -r requirements.txt
```

或者不污染 ComfyUI 环境：

```bash
uv run --offline --no-project \
  --with "rea-video[ocr,asr] @ git+https://github.com/JuneYaooo/rea-video.git@a124acd4a02b3a9b72b9e7dedca5d1bfcda2a7b8" \
  --with pillow --with piper-tts python -u engine/rea_remake.py --help
```

### 4.6 启动 H3 引擎

```bash
bash engine/start_engine.sh > engine/logs/engine.log 2>&1 &
curl -s http://127.0.0.1:1022/system_stats
```

可覆盖的环境变量：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `COMFY_DIR` | `/g/comfyui` | ComfyUI 根目录 |
| `COMFY_MODELS` | `$COMFY_DIR/models` | 模型根目录 |
| `REA_ENGINE_PORT` | `1022` | 引擎端口 |
| `REA_LISTEN` | `127.0.0.1` | 绑定地址 |
| `REA_RESERVE_VRAM` | `6` | 给系统留的显存 GiB |
| `REA_SPARK` | `0` | 1 = 额外加载 Spark-H3（实测更慢，默认关） |

启动参数里的 `--disable-fast-disk --disable-async-offload --disable-pinned-memory --disable-cuda-malloc` 是这台机器上稳定跑通的组合，别随手删。

对应的 Python 侧地址用 `REA_ENGINE_URL`（默认 `http://127.0.0.1:1022`），视觉端口用 `REA_VISION_PORT`（默认 18095）。

---

## 5. 跑一条片子

```bash
# 1) 拆解参考片
python engine/rea_remake.py analyze out/demo reference.mp4

# 2) 商品理解 + 动作规划 + 构图（1–3 张商品图）
python engine/rea_remake.py plan out/demo product1.png product2.png product3.png

# 改了提示词/商品事实后重建（保留已渲染片段，历史记录改名 .superseded）
python engine/rea_remake.py plan out/demo product1.png product2.png product3.png --reprompt

# 3) 逐段生成 + 拼接 + 配音 + 验收
python engine/rea_remake.py render out/demo

# 只重跑某几个 H3 调用，之后再拼接
python engine/rea_remake.py render out/demo --only G3,G4 --no-finish
python engine/rea_remake.py finish out/demo

# 明确跳过验收（只用于调试，不会报告成功）
python engine/rea_remake.py render out/demo --skip-gate
```

产物都在 `out/demo/remake/`：

| 文件 | 内容 |
| --- | --- |
| `remake.mp4` | 成片（1080×1920, 30 fps, AAC） |
| `remake.with-narration.mp4` | 混音中间产物 |
| `h3-audio.wav` | H3 生成音轨拼成的底噪 |
| `narration.json` / `narration.wav` | 改写文案、逐句起止、ASR 回读、语速对齐 |
| `actions.json` | 每镜动作、是否改写、商品事实 |
| `contract.json` | 商品部件、缺失部件、禁用动作 |
| `acceptance.json` | 验收结论与失败项 |
| `review.json` | 逐镜复核记录（按镜头帧哈希） |
| `review/sheet-*.jpg` | 复核用抽帧拼图 |

---

## 6. 验收门

`engine/check_film.py` 做四件事：

1. **结构**：帧数、fps、时长、音轨、完整解码
2. **口播**：只允许商品事实里出现的颜色/部件；出现 `lace/strap/box/insert` 等缺失部件、未证实卖点、未核对数字 → 拒绝
3. **契约**：原片需要系带/拆盒而商品没有 → `actions` 必须 `adapt`，且改写后的动作不能又把这些部件演出来
4. **逐镜人工复核**：按镜头中帧哈希记录，重新出片后未复核或帧变了的镜头都会重新变成 missing

复核流程：

```bash
python engine/check_film.py out/demo --sheet
# 打开 out/demo/remake/review/sheet-01-11.jpg 等，逐镜看
python engine/check_film.py out/demo --review 1=pass 2=pass 3=fail:still-white-sneakers
```

结论写进 `acceptance.json`：`accepted` 或 `rejected`，退出码 0/2。`finish()` 里门不过会抛 `RuntimeError`，不会把坏片当成功。

为什么不用视觉模型判商品身份：实测 Qwen2.5-Omni-7B 对“这是不是同一个商品”的强制选择在明显白鞋画面上仍答 `same / high`。模型适合读属性、不适合下判决，所以判决交给抽帧复核 + 确定性契约。

---

## 7. 配音与声音

- 原片口播先被 ASR 成文字，程序读原意后按**当前商品事实**重写（默认 3 句，50–60 词，源语言）
- 只允许事实里有的内容；`audit()` 拦未知颜色、未证实卖点、未核对数字
- 用 piper `en_US-lessac-medium` 合成，按原片语速自动做两轮时长校正，再按原片句子起点排布
- 混音：H3 生成的现场音作为底噪 −20 dB，口播出现时侧链压低，整体 loudnorm 到 −16 LUFS
- 音轨回读：成片重新 ASR，与文案比对，准确率 < 0.85 拒绝；`performance_similarity` 记录与原声的包络相似度

**做不到的**：这套 H3 插件的 Director 节点没有音频参考输入（`inputs` 只有 prompt / duration / ref_video / ref_images），所以不能把逐镜 WAV 塞回 H3 做口型条件，也不做人声克隆。现状是“重写口播 + 保留生成音床”，不是原声复刻。

---

## 8. 队列与网页

```bash
python engine/service.py
# 打开 http://127.0.0.1:8099
```

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `REA_HOST` | `127.0.0.1` | 监听地址 |
| `REA_PORT` | `8099` | 网页端口 |
| `REA_TOKEN` | 空 | 设了就要求 `Authorization: Bearer`（页面本身还不带 token，非本机访问请用反向代理加认证） |
| `REA_QUEUE` | `engine/state/queue.json` | 队列文件（测试用隔离） |

接口：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/health` | 队列长度、引擎可达/忙、等待中与运行中的任务 |
| GET | `/api/jobs` | 任务列表 |
| GET | `/api/jobs/<id>/log` | 日志尾部 8 KB |
| POST | `/api/jobs` | 提交 `{source, photos[], out, narration, skip_gate}` |
| POST | `/api/jobs/<id>/cancel` | 取消排队中的任务 |
| POST | `/api/jobs/<id>/reorder` | `{before: <id>}` 在自己等待位内重排 |
| POST | `/api/jobs/<id>/resume` | `{confirm:false}` 只对账，`{confirm:true}` 才续跑 |
| GET | `/api/media?id=<id>` | 播放该任务的成片 |

断点续跑规则：服务启动时把 `running` 标记成 `interrupted`，并用 ComfyUI `/queue`、`/history` 对账出 `not_started / still_running / finished / partial / unknown`。只有 `not_started`（确认没提交过）允许确认后续跑；`unknown` 一律拒绝，避免重复提交同一段 H3。

队列自检（不需要 GPU）：

```bash
python engine/test_jobs.py
```

---

## 9. 实测基线（RTX 3090 24 GB，2026-11）

| 项 | 数值 |
| --- | --- |
| 8 个 H3 调用（31 镜，124 帧/段，4 步） | 1089.0 s，单段 134.6–141.1 s |
| 整轮 render（含下载/编码/拼接/配音/门） | 1213 s |
| 峰值显存 | 21,871 MiB |
| 成片 | 1046 帧 / 34.867 s / 1080×1920 / 30 fps / AAC |
| 拼接与混音 | ≈ 100–110 s |
| 口播 ASR 回读 | 0.9744（新版文案） |

这些数字是这套权重、4 步 turbo、480×864 下的结果；换分辨率、步数、显卡都要重测，不要拿短片段外推长片。

---

## 10. 已知限制

1. **没有音频参考条件**：见第 7 节，口型不跟随重写口播，只能做音床 + 旁白。
2. **商品身份靠人工复核兜底**：视觉模型判身份不可信，这是刻意的设计取舍。
3. **单机单 worker**：`engine/jobs.py` 一次只跑一个 H3 任务；多卡需要改 worker。
4. **网页不带 token**：本机用可以，暴露到局域网前必须加认证。
5. **不自动挑商品图**：1–3 张图由调用方给；商品事实来自 plan 阶段对图的解读。
6. **只验证了 Windows + Git Bash**：Linux 需要改 `start_engine.sh` 里的 `cygpath` 和路径。

---

## 11. 排障

| 现象 | 原因 / 处理 |
| --- | --- |
| `Address already in use` | 1022 已被占用：`netstat -ano \| grep 1022`，先停旧进程 |
| `clip_name ... not found` | 缺 AWQ 编码器：按 4.2 下载，或设 `REA_CLIP_NAME` 指向已有文件 |
| 构图报 `nodes missing` | 三个自定义节点没装/版本不对，按 4.1 对齐 commit |
| 显存不足 | 降 `REA_RESERVE_VRAM` 不等于能省显存；关掉别的占卡进程；确认 `--disable-fast-disk` 等参数在 |
| 视觉服务起不来 | `REA_VISION_EXE` / `REA_VISION_VENDOR` / `REA_VISION_MODELS` 三处路径；端口 18095 被占 |
| 门一直缺 review | 正常：新片每个镜头都要复核一次，`--sheet` 看图后 `--review` 记录 |
| ASR 准确率低 | 换更干净的 TTS 文本（短句、少专名），或确认 sensevoice 模型目录名完全一致 |

---

## 12. 目录

```
armA_seg01_graph.json      H3 参考图生视频模板（构图起点）
requirements.txt           Python 依赖（含上游 pin）
engine/
  rea_remake.py            analyze / plan / render / finish 主流程
  auto_plan.py             画面理解、动作规划、构图
  contract.py              商品部件契约（缺件→改动作）
  check_film.py            验收门与逐镜复核
  narration.py             文案改写、TTS、语速对齐、混音、ASR 回读
  jobs.py                  队列、取消、重排、断点对账
  service.py               网页与 JSON API
  local_vision.py          本地 llama.cpp 视觉服务
  launch_engine.py         ComfyUI 启动包装
  start_engine.sh          引擎启动脚本
  test_jobs.py             队列自检
  web/index.html           网页界面
```

---

## 13. 上游与许可

- 上游流程：[JuneYaooo/rea-video](https://github.com/JuneYaooo/rea-video)，使用时保持其许可与署名
- 模型：[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)，遵循 MiniMax-H3 模型许可
- H3 插件：ComfyUI_MiniMaxH3_Director（MIT，以仓库为准）
