# Data Engine V1 方案（Manifest 联邦 + 数据源注册表 + 统一样本 Schema）

## 1. 背景与目标

Data Engine V1 的目标是为文档解析训练数据构建一套可运行、可扩展、可追溯的增量式数据引擎。它服务于“数据不会一次性到位、而是按类别分批持续进入”的现实场景，因此系统设计必须优先支持增量处理、断点续跑、跨批次统一采样和稳定的样本追溯。

本方案明确不使用数据库。原因不是临时简化，而是针对当前数据形态的主动架构选择：

- 数据天然以目录和批次组织，数据库并不是事实来源。
- 数据源可能位于多块硬盘、不同挂载点甚至不同机器路径，数据库难以成为稳定锚点。
- 某一批次数据可能整体失效或需要下线，直接隔离目录比维护全局表更简单可靠。
- 大规模预训练数据常用“文件系统 + manifest”方式管理，增量扩展和版本冻结更自然。

因此，V1 采用以下总体原则：

- 不使用数据库。
- `sources.yaml + manifests` 是系统唯一事实来源。
- 以 `source / category / batch` 管理数据物理位置和逻辑归属。
- 以统一样本 Schema 作为中间表示，承载多模型输出、比较、修正和导出。
- 以统一状态机作为任务编排基线，保证自动流程、人工流程和 QA 流程不会脱节。

V1 是一个可运行 MVP，而不是最终生产版。重点是先把完整闭环搭起来，并确保后续可以继续向生产级实现演进。

## 2. 总体架构

Data Engine V1 采用三层架构：

1. `Source Registry`
   - 负责记录物理数据源的位置和逻辑身份。
   - 提供 `source_id -> root_path` 的稳定映射。

2. `category / batch` 目录组织
   - 按类别与批次管理增量数据。
   - 每个批次是最小处理和回滚单元。

3. `manifest` 联邦
   - 每个批次保存自己的阶段结果和状态。
   - 全局任务在运行时聚合多个 manifest，而不是依赖全局数据库表。

在此基础上，系统还有两个核心约束：

- `Unified Sample Schema`
  - 所有模型输出必须先转换到统一中间表示，再进入比较、修正和导出。

- `Unified State Machine`
  - 所有处理过程都围绕同一套样本状态流转，不允许自动流程和人工流程各自维护独立状态。

## 3. 存储与路径设计

### 3.1 数据源注册表

项目根目录维护一个 `sources.yaml`，作为数据源地图：

```yaml
sources:
  - id: "disk_01"
    root_path: "/mnt/external_sata_1/finance_data"
    category: "finance"
    enabled: true
  - id: "disk_02"
    root_path: "/media/user/backup_disk/japanese_news"
    category: "japanese_news"
    enabled: true
  - id: "remote_node_c"
    root_path: "/home/ubuntu/data/textbooks"
    category: "textbook"
    enabled: true
```

字段含义：

- `id`
  - 稳定的逻辑数据源 ID，也是 manifest 中引用物理资产的锚点。
- `root_path`
  - 当前物理挂载路径或本地路径。
- `category`
  - 该数据源默认归属的类别。
- `enabled`
  - 是否参与当前扫描、聚合和导出。

### 3.2 逻辑路径与物理路径

Manifest 不保存绝对路径，只保存：

- `source_id`
- `category`
- `batch_id`
- `relative_path`

运行时再解析为真实文件路径：

```text
physical_path = sources[source_id].root_path / relative_path
```

例如：

- `source_id = disk_01`
- `relative_path = batch_20260510/page_images/page_000123.jpg`

这样即使硬盘挂载点改变，也只需修改 `sources.yaml`，无需重写任何历史 manifest。

### 3.3 批次目录结构

每个 source 根目录下按 batch 组织：

```text
<root_path>/
  batch_<batch_id>/
    raw_input/
    page_images/
    elements/
    renders/
    manifests/
      ingest.parquet
      page_sampling.parquet
      element_sampling.parquet
      cmcv.parquet
      refine.parquet
      export_view.parquet
    artifacts/
      config_snapshot.yaml
      stats.json
    .engine_meta.yaml
```

说明：

- `raw_input/`
  - 原始输入目录，兼容 PDF、单张图片和文件夹形式的图片集。
- `page_images/`
  - 页面级图像资源。
- `elements/`
  - 元素级切分产物，例如文本块、表格块、公式块。
- `renders/`
  - Judge-and-Refine 阶段的渲染结果。
- `manifests/`
  - 每个阶段各自独立的 manifest 文件。
- `artifacts/`
  - 该批次产生的统计和配置快照。
- `.engine_meta.yaml`
  - 用于 `register` / `scan` 自动识别的元信息文件。

补充约束：

- `page_images/` 是物理资产保护区。
- 一旦 ingest 完成，`page_images/` 中的页面图像只允许被读取，不允许被后续阶段覆盖、重写或原地修改。
- `elements/`、`renders/` 以及其他派生产物必须写入新的目标目录，不能回写到 `page_images/`。

## 4. 统一样本 Schema

必须先定义统一中间表示，否则多模型结果难以比较、难以重放，也难以保证不同阶段之间的数据语义一致。

V1 规定统一样本最少包含以下结构：

- `page_image`
  - 页面图像引用。
- `block_list`
  - 页面块级对象列表，覆盖 `layout / text / formula / table`。
- `reading_order`
  - 页面或块内的阅读顺序信息。
- `table_structure`
  - 表格的结构化表达，至少能表示单元格、行列关系和 span。
- `formula_spans`
  - 公式区域、bbox、内容以及块级/行内属性。
- `text_spans_with_bbox`
  - 文本 span 及其 bbox，必要时带行级或段级归属关系。

### 4.1 UnifiedSampleRecord

建议统一记录结构如下：

- `sample_id`
- `source_id`
- `category`
- `batch_id`
- `input_type`
- `original_ext`
- `page_id`
- `task_type`
- `relative_path`
- `page_image`
- `block_list`
- `reading_order`
- `table_structure`
- `formula_spans`
- `text_spans`
- `bbox`
- `cluster_id`
- `difficulty`
- `annotation_source`
- `stage_status`
- `process_log`
- `data_version`
- `schema_version`
- `threshold_version`
- `is_active`

约束如下：

- 所有模型输出必须先归一化，再写入统一 schema。
- 所有比较逻辑只读取统一 schema 中的标准化字段。
- 所有修正逻辑都作用于统一 schema，不直接依赖模型私有输出格式。
- 所有导出结果必须能回溯到对应的 `schema_version`。
- 所有导出结果必须能回溯到原始输入类型和原始文件后缀。

新增字段说明：

- `input_type`
  - `pdf` 或 `image`，用于标识该样本来自 PDF 还是图片输入。
- `original_ext`
  - 原始文件后缀，例如 `.pdf`、`.png`、`.jpg`、`.jpeg`、`.webp`。
  - 该字段用于排查解码、转码、归一化和上游数据接入问题。

### 4.2 标准化层

在 CMCV 前，所有模型输出先通过标准化层，转换为统一表示：

- `NormalizedTextPrediction`
- `NormalizedTablePrediction`
- `NormalizedFormulaPrediction`
- `NormalizedLayoutPrediction`

标准化层职责：

- 抹平不同模型输出格式差异。
- 保证一致性比较只发生在统一表示上。
- 保证同一样本在不同阶段可回放、可追溯、可复查。

这意味着所有比较必须先经过统一 schema 和标准化层，不能直接比较模型原始输出。

## 5. 状态机与任务编排

为了避免人工流程和自动流程脱节，V1 规定每个样本必须经过同一条主状态流转：

```text
ingested -> embedded -> clustered -> scored -> inferred -> compared -> bucketed -> annotated -> qaed -> released
```

各状态含义如下：

- `ingested`
  - 样本已注册并生成基础记录。
- `embedded`
  - 已完成 embedding 提取。
- `clustered`
  - 已完成簇分配。
- `scored`
  - 已完成采样分数或前置质量分数计算。
- `inferred`
  - 已完成多模型推理。
- `compared`
  - 已完成模型间一致性计算。
- `bucketed`
  - 已完成 `easy / medium / hard / invalid` 分桶。
- `annotated`
  - 已完成自动标注、自动修正或人工标注回填。
- `qaed`
  - 已通过 QA 或完成 QA 结论写回。
- `released`
  - 已进入最终训练集导出视图。

### 5.1 状态机规则

- `stage_status`
  - 仅表示当前主状态。
- `process_log`
  - 追加记录阶段版本，例如 `ingest_v1`、`cmcv_v1`、`refine_v1`。
- 失败态
  - 允许以 `*_failed` 形式附着在主阶段上，例如 `compared_failed`、`annotated_failed`。
- 自动流程、人工流程、QA 流程
  - 必须写回同一状态机，不允许另起一套外部状态系统。

### 5.2 幂等与续跑

- 每个阶段只处理“当前阶段未完成且上游已完成”的记录。
- `--resume` 通过 manifest 中的 `stage_status` 和 `process_log` 实现。
- 已完成更后续状态的样本，不允许被前序阶段误覆盖或回退。

### 5.3 分片处理的确定性分配

当 CLI 支持 `--shard-id` 与 `--num-shards` 时，V1 采用确定性分片，而不是依赖中心化任务分配。

推荐规则：

```text
is_my_task = int(sample_id, 16) % num_shards == shard_id
```

实现约束：

- 分片归属仅依赖 `sample_id` 和 `num_shards`
- 不依赖 master 节点动态派发任务
- 相同输入和相同分片参数下，分片结果必须稳定可复现
- 样本分配必须尽量均匀，避免局部热点

说明：

- 如果后续需要升级为一致性哈希，实现必须保持“同一 sample_id 在相同分片配置下归属稳定”这一性质
- V1 默认使用简单取模逻辑即可

### 5.4 Parquet 并发写约束

虽然 Parquet 适合做联邦查询和阶段落盘，但 V1 在多进程环境下不应采用“多个进程同时追加同一个 parquet 文件”的方式。

推荐策略：

- 每个 worker 或 shard 先写独立临时文件，例如：
  - `manifests/temp_shard_001.parquet`
  - `manifests/temp_shard_002.parquet`
- 当前阶段全部 worker 完成后，再执行一次显式 merge
- merge 结果写入正式阶段产物，例如 `cmcv.parquet`、`page_sampling.parquet`

实现约束：

- 不依赖 parquet 原地追加作为主写入策略
- merge 操作必须是显式且可重跑的
- merge 前后都要保留 shard 级来源信息，便于失败排查
- 若使用 `pandas.concat` 或等价逻辑进行合并，最终写盘前应执行去重和 schema 对齐检查

## 6. 六阶段流程

V1 主流程保持为六阶段，加上两个入口工具：

- `register`
- `scan`
- `ingest`
- `page-sample`
- `element-sample`
- `cmcv`
- `hardcase-refine`
- `export`

### 6.1 register

用途：

- 把新的物理数据源加入 `sources.yaml`。
- 扫描目标目录是否存在 `.engine_meta.yaml` 或已有 manifest 结构。

输入：

- `--path`
- 可选 `--source-id`
- 可选 `--category`

输出：

- 更新 `sources.yaml`

约束：

- 不搬移数据。
- 只建立逻辑映射。

### 6.2 scan

用途：

- 扫描所有已注册 source 的在线状态、batch 数量和 manifest 完整性。

输出：

- source inventory 视图或统计报告。

### 6.3 ingest

用途：

- 从某个 source 的某个 batch 中读取原始输入，并生成页面级记录。

输入：

- `--source-id`
- `--batch`

原始输入目录约定为：

```text
batch_<batch_id>/
  raw_input/
    book_A/
      001.jpg
      002.jpg
    doc_B.pdf
    single_photo.png
```

说明：

- `ingest` 递归扫描 `raw_input/`。
- 输入既可以是单个 PDF，也可以是单张图片，也可以是“一个文件夹下的一组图片”。
- 如果输入是文件夹内图片，`relative_path` 必须保留文件夹层级，避免同名文件冲突，并保留原始来源结构。

处理逻辑必须根据文件后缀自动分流。

#### PDF 输入

识别条件：

- 文件后缀为 `.pdf`

处理动作：

- 拆分页面
- 渲染页面图像
- 提取元数据
- 如果 PDF 内嵌文本层存在，则将其作为前置参考信息写入 ingest 产物

ID 规则：

- `page_id = hash(pdf_content) + page_no`

约束：

- 同一个 PDF 的不同页必须生成稳定且可复现的 `page_id`
- PDF 文本层只作为前置参考，不直接替代后续标准化标注结果
- 生成页面图像后，必须对页面图像文件计算 `SHA256`，并写入 `ingest.parquet`，作为后续完整性校验基线

#### 图片输入

识别条件：

- 文件后缀属于图片类型，例如 `.jpg`、`.jpeg`、`.png`、`.webp`

处理动作：

- 格式转码
- 统一转为标准无损或高质量 JPG/PNG
- 图像归一化
- 统一 DPI 或等价分辨率元数据

ID 规则：

- `page_id = hash(image_content)`

约束：

- 图片输入默认按“单页样本”处理
- 文件夹中的多张图片视为一组顺序输入，但每张图片仍拥有独立 `page_id`
- 若后续需要表达“一个文件夹代表一本书”，应通过额外文档级元数据关联，而不是复用同一个 `page_id`
- 如果 ingest 阶段执行了 DPI 归一化、缩放或其他几何变换，必须把原始尺寸、归一化后尺寸和缩放比例写入 `source_metadata`
- 归一化完成后的页面图像文件必须计算 `SHA256` 并写入 `ingest.parquet`

输出：

- `ingest.parquet`

状态推进：

- 进入 `ingested`

`ingest.parquet` 至少要写入以下与输入形态相关的字段：

- `input_type`
- `original_ext`
- `relative_path`
- `page_id`
- `page_image_sha256`
- 可选的 `source_metadata`
  - 对 PDF 可包含文本层、页数、原始元信息
  - 对图片可包含原始尺寸、归一化后尺寸、缩放比例、色彩模式、原始 DPI、归一化 DPI

补充约束：

- `element-sample` 产生的 bbox 默认是基于归一化后的页面图像
- 如果后续需要回溯到原始高分辨率图像进行 OCR 精修或人工复核，必须依赖 `source_metadata` 中记录的缩放比例完成坐标换算
- 后续阶段如需校验页面图像未被破坏，应优先比对 `page_image_sha256`

### 6.4 page-sample

用途：

- 在 category 视图下完成页级 embedding、聚类和初步评分。

输入：

- `--category`

处理逻辑：

- 聚合同 category 下所有在线 source 的 `ingest.parquet`
- 计算页级 embedding
- 执行页级聚类
- 抽取 probe 样本并做页级 CMCV 难度感知评分

状态推进：

- `embedded -> clustered -> scored`

输出：

- 各 batch 的 `page_sampling.parquet`

### 6.5 element-sample

用途：

- 对页级候选样本执行 layout detection 和元素切分。

处理对象：

- `layout`
- `text`
- `formula`
- `table`

输出：

- 块级对象
- 阅读顺序
- 公式 span
- 文本 span
- 表格结构
- `element_sampling.parquet`

该阶段负责补全统一 schema 中的结构字段。

### 6.6 cmcv

用途：

- 执行 Cross-Model Consistency Verification。

处理逻辑：

- 对每个样本运行三模型推理
- 将结果标准化为统一表示
- 计算两两一致性分数
- 按规则分桶

难度规则固定为：

- `easy`
  - 目标模型与至少一个外部模型一致。
- `medium`
  - 两个外部模型一致，但目标模型不一致。
- `hard`
  - 三个模型两两都显著不一致。

状态推进：

- `inferred -> compared -> bucketed`

输出：

- `cmcv.parquet`

### 6.7 hardcase-refine

用途：

- 只处理 `hard` 样本中的高价值对象，优先是公式和表格，必要时可包括文本。

处理逻辑：

- 公式与表格走 render-compare refine
- 比较原图与渲染图
- 定位错误
- 迭代修正
- 重新渲染并再次验证

输出状态：

- `auto_fixed`
- `needs_expert`
- `invalid`

状态推进：

- 成功修正或人工回填后进入 `annotated`

### 6.8 export

用途：

- 跨 source、跨 category 生成最终训练集逻辑视图。

特点：

- 不复制底层文件
- 只导出 `jsonl` 或 `parquet`
- 记录 `source_id + relative_path` 或其解析快照

状态推进：

- QA 完成后进入 `qaed`
- 导出完成后进入 `released`

## 7. 增量聚类策略

V1 采用 category 级“锚点中心”策略，而不是每次新批次到来就全量重聚类。

策略如下：

1. 首次处理某 category 时
   - 建立初始聚类中心。

2. 新 batch 到来时
   - 先读取已有中心点。
   - 使用最近邻将新样本分配到现有簇。

3. 新样本离所有中心点都过远时
   - 视为出现新分布。
   - 为其创建新簇。

4. 所有聚类结果必须记录
   - `centroid_version`

约束：

- `category` 是聚类主边界。
- `source` 是物理载体，不是建模边界。
- 后续不同 source 的同类数据必须共享同一套 category 级簇体系。

工程实现建议：

- 聚类索引文件与 manifest 分开存储，不写入各阶段 parquet
- 中心点、近邻索引或 Faiss 索引统一放在 `artifacts/` 或独立的 engine state 目录
- manifest 中只保留轻量引用信息，例如：
  - `cluster_id`
  - `centroid_version`
  - 可选的 `index_snapshot_id`

原因：

- 在千万到亿级样本规模下，Faiss 或等价索引文件可能非常大
- 将索引与 manifest 解耦，便于增量归类、版本切换和跨批次复用
- 这样可以保证 manifest 仍然是轻量、可联邦查询、可快速 merge 的状态文件

## 8. 公共接口

V1 至少定义以下公共记录类型。

### 8.1 UnifiedSampleRecord

- `sample_id`
- `source_id`
- `category`
- `batch_id`
- `input_type`
- `original_ext`
- `page_id`
- `task_type`
- `relative_path`
- `page_image`
- `block_list`
- `reading_order`
- `table_structure`
- `formula_spans`
- `text_spans`
- `bbox`
- `cluster_id`
- `difficulty`
- `stage_status`
- `process_log`
- `annotation_source`
- `data_version`
- `schema_version`
- `threshold_version`
- `is_active`

### 8.2 PredictionRecord

- `sample_id`
- `model_name`
- `model_role`
- `normalized_output`
- `raw_output_ref`
- `schema_version`
- `render_relative_path`
- `inference_meta`

### 8.3 ConsistencyRecord

- `sample_id`
- `task_type`
- `scores`
- `tier`
- `threshold_version`
- `backend_triplet`

### 8.4 AnnotationRecord

- `sample_id`
- `annotation_source`
- `final_label`
- `review_trace`
- `qa_status`

### 8.5 ExpertQueueRecord

- `sample_id`
- `source_id`
- `category`
- `batch_id`
- `priority`
- `failure_reason`
- `judge_trace`
- `recommended_action`

## 9. CLI 约定

核心命令约定如下：

```text
data_engine register --path <path> [--source-id <id>] [--category <cat>]
data_engine scan
data_engine ingest --source-id <id> --batch <batch_id>
data_engine page-sample --category <cat>
data_engine element-sample --category <cat>
data_engine cmcv --category <cat>
data_engine hardcase-refine --category <cat>
data_engine export --export-policy <yaml>
```

统一支持以下可选参数：

- `--resume`
- `--batch`
- `--shard-id`
- `--num-shards`
- `--config`

说明：

- `register` 和 `scan` 是系统级入口。
- `ingest` 通常面向单 batch。
- `page-sample`、`element-sample`、`cmcv`、`hardcase-refine` 通常在 category 视图上执行。
- `export` 负责生成全局逻辑训练视图。

## 10. 测试与验收标准

### 10.1 功能测试场景

- 更新 `sources.yaml` 中某个 `root_path` 后，历史 manifest 无需修改仍可正常解析文件。
- 新 source 注册后可以被 `scan` 正确发现。
- 同 category、不同 source 的多个 batch 能被正确聚合处理。
- 多模型输出经过标准化后，能稳定映射到统一 schema。
- CMCV 能正确给出 `easy / medium / hard` 分桶。
- `hardcase-refine` 能覆盖以下路径：
  - 首轮修正成功
  - 多轮修正后成功
  - 渲染失败进入专家队列
  - 达到最大轮次仍失败
- `export` 生成逻辑视图而不复制原始文件。
- `page_images/` 在 ingest 完成后保持只读语义，后续阶段只产生派生文件，不改写原始页图。
- `ingest.parquet` 中记录的 `page_image_sha256` 能用于校验页面图像完整性。

### 10.2 幂等与恢复测试

- 重复执行同一阶段时，不重复加工已完成记录。
- source 离线恢复后，`--resume` 能继续未完成流程。
- 已进入后续状态的样本不会被前序阶段覆盖或回退。
- 多进程执行同一阶段时，先写 shard 临时 parquet，再 merge 成正式 manifest，且 merge 结果不重复、不丢记录。
- 相同 `sample_id`、`num_shards`、`shard_id` 配置下，多次运行得到完全一致的任务分片结果。

### 10.3 路径切换与离线 source 测试

- 某 source 挂载点变化后，仅修改 `sources.yaml` 即可恢复读取。
- 某 source 离线时，任务能跳过该 source 并输出明确告警。
- 某 source 被禁用后，不再参与聚合与导出。

### 10.4 Schema / Version 可追溯测试

- 每条导出样本都能追溯到：
  - `source_id`
  - `batch_id`
  - `relative_path`
  - `schema_version`
  - `threshold_version`
  - `process_log`
- 聚类结果能通过 `centroid_version` 回溯。
- 图片样本若经过归一化，能够通过 `source_metadata` 中的尺寸和缩放比例，将 bbox 正确映射回原始图像坐标系。
- 页面图像若被意外篡改或覆盖，能够通过 `page_image_sha256` 检测出来。

### 10.5 统计信息测试

- 每个 batch 完成关键阶段后，`artifacts/stats.json` 至少包含样本数量统计和难度分布直方图。
- 难度分布直方图至少覆盖：
  - `easy`
  - `medium`
  - `hard`
  - `invalid`
- 统计结果应能帮助快速判断某批次是普通样本为主，还是高难样本为主。

## 11. 后续衔接建议

本方案文档写入后，下一步实现建议按以下顺序推进：

1. 新增 `sources.yaml.example`
2. 新增 `data_engine/` Python 包骨架
3. 新增 `configs/` 默认策略配置
4. 新增 `tests/` 下的状态机与 schema 单测

这些内容都应以本文件作为唯一规格基线，不再另起冲突版本。

### 11.1 首批 Milestone 划分

建议按以下三个 Milestone 推进首轮开发：

#### M1: 数据底座（Data Foundation）

目标：

- 建立数据源注册与统一记录模型，先打通“物理资产如何被系统识别和接入”。

范围：

- 实现 `sources.yaml` 的解析逻辑
- 定义 `UnifiedSampleRecord` 的 Pydantic 模型
- 编写 `register` 和 `scan` 命令
- 跑通物理硬盘或挂载目录的逻辑接入

交付结果：

- 能从 `sources.yaml` 正确加载和校验 source 配置
- 能把新路径注册为逻辑数据源
- 能扫描 source 在线状态、batch 结构和 manifest 完整性
- 统一 schema 模型可被后续阶段直接复用

#### M2: 资产标准化（Asset Standardizer）

目标：

- 建立 ingest 基线，把原始输入转成稳定、可追溯的页面级资产。

范围：

- 编写 `ingest` 逻辑
- 支持 PDF 拆页
- 支持图片归一化
- 实现内容哈希 ID 生成器
- 生成 `page_id`、`page_image_sha256`、`source_metadata`

交付结果：

- 能递归扫描 `raw_input/`
- 能按后缀自动区分 PDF 和图片处理逻辑
- 能输出 `ingest.parquet`
- 能生成稳定的内容哈希 ID 和页面图像校验信息

#### M3: 状态流转监控（Pipeline Watcher）

目标：

- 在完整后续阶段尚未全部实现前，先提供一个统一的状态观察入口。

范围：

- 实现 `data_engine status` 命令
- 聚合所有 batch 的 `stage_status`
- 输出各 category、各 batch 的进度概览
- 支持查看样本数、已完成阶段数和待处理数量

交付结果：

- 能快速查看全局进度
- 能发现哪些 batch 卡在 `ingested`、`bucketed` 或 `annotated`
- 为后续 `page-sample`、`cmcv`、`hardcase-refine` 的调度和排障提供基础观测能力

#### M4: Web Console

目标：

- 在无数据库前提下，为 Data Engine 提供可视化控制台，覆盖数据地图、Hard Case 复核和导出前 QA 抽检。

范围：

- 实现基于文件聚合的 Web 后端
- 实现 Dashboard 首页“数据地图（The Map）”
- 实现 Hard Case 对比复核页面
- 实现导出前随机抽检页面
- 实现有限写操作：复核结论、QA 标记、抽检结果写回

交付结果：

- 能在不依赖数据库的前提下，聚合 `sources.yaml`、`manifests/*.parquet` 与 `artifacts/stats.json`
- 能提供全局 source / batch / stage / difficulty 的可视化视图
- 能支持 `hard` 样本的图像级人工复核
- 能支持导出前样本随机抽检与结果记录

### 11.2 Web Console 规划

Web Console 是 Data Engine 的正式子系统，但不替代 CLI 主流程。

定位：

- `M1/M2/M3` 继续以 CLI 和 manifest 为主
- `M4` 引入 Web Console
- Web 首版以开发者使用为主，但必须覆盖真实复核、巡检和抽检场景

数据来源：

- `sources.yaml`
- 各 batch 的 `manifests/*.parquet`
- 各 batch 的 `artifacts/stats.json`
- 各 batch 的 `.engine_meta.yaml`
- `page_images/`
- `renders/`

实现原则：

- Web 后端通过运行时聚合文件状态源工作
- 可以使用内存缓存或本地只读快照提升响应速度
- 缓存不是事实来源，事实来源仍然是 `sources.yaml + manifests + stats`
- 首版不允许在 Web 中直接触发全流程重跑

### 11.3 数据地图（The Map）

`数据地图（The Map）` 是 Web Console 的首页视图，也是 Dashboard 的第一屏。

目标：

- 解决数据分散在不同硬盘、不同挂载点下时的“全局掌控感缺失”问题

数据来源：

- `sources.yaml`
- 各 source 下各 batch 的 `artifacts/stats.json`
- 必要时补充聚合后的 `status` 视图或 `ingest.parquet`

主要展示内容：

- 每个 `source` 的在线 / 离线状态
- 每个 `source` 的物理挂载路径
- 每个 `source` 下的 batch 数量
- 已处理页面数
- `easy / medium / hard / invalid` 分桶比例
- 最近一次扫描或统计更新时间

核心价值：

- 让用户第一眼知道哪些 source 在线
- 快速判断数据主要分布在哪些硬盘或目录
- 快速识别哪些 batch 偏 `hard`、哪些 batch 偏 `easy`
- 为 Hard Case 复核和导出前 QA 提供入口导航

## 12. 开发落地补充约束

### 12.1 物理资产保护区

- `page_images/` 是全流程基础资产目录，默认按只读语义管理。
- ingest 阶段生成页面图像后，必须为每个页面图像记录 `page_image_sha256`。
- 后续 `elements/`、`renders/`、QA 截图或其他衍生产物都必须写入新文件，禁止覆盖 `page_images/` 中原图。

### 12.2 分片处理规则

- 所有支持 `--shard-id` 的阶段默认使用确定性取模逻辑进行任务领取。
- 推荐实现：

```text
int(sample_id, 16) % num_shards == shard_id
```

- 这样每个节点只需要读取 manifest，就能独立判断自己该处理哪些样本。

### 12.3 统计信息的实时脉搏

- 每个 batch 的 `artifacts/stats.json` 不只记录总量，也要记录难度分布直方图。
- 最少包含：
  - 总样本数
  - 有效样本数
  - `easy / medium / hard / invalid` 分布
- 这些统计用于快速判断当前批次的训练价值和专家标注压力，并为后续优先级调整提供依据。
