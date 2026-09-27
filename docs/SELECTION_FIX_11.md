# Pathflow Studio 0.9.1 · Selection and discovery fix

The 0.9.0 release added a separate joint-task workspace but did not expose it through the existing design selectors. Its own dataset dropdown was populated only from prepared datasets, so a fresh installation could display an empty dropdown. ResNet-50 was implemented in the backend but had no explicit architecture/preset control. These were UI integration omissions.

This fix adds BUS-UCLM and the synthetic joint-task dataset to the main Design experiment dataset selector, and ResNet-50 to the architecture family selector. Selecting either joint-task dataset or ResNet-50 opens the joint configuration form, carrying the experiment name. The original MLP/CNN configuration and validators remain unchanged.

The joint dataset catalog always includes BUS-UCLM and the synthetic engineering dataset, even before local preparation. Each entry has an explicit readiness state. Create stays disabled until the chosen dataset is prepared; its adjacent preparation button handles the selected built-in dataset. Imported datasets appear alongside the built-ins.

The architecture selector explicitly names ResNet-50 with its decoder and classifier. Presets apply actual configuration values:

| Preset | Resolution | Decoder width | Initialization | Dataset | Evaluation |
|---|---:|---:|---|---|---|
| Medical baseline | 256 | 32 | Random | BUS-UCLM | Full split |
| ImageNet fine-tuning | 256 | 32 | Pretrained V2 | BUS-UCLM | Full split |
| Engineering check | 64 | 16 | Random | Synthetic | 3 cases/split |

Both medical presets use Adam, learning rate 1e-4, weight decay 1e-4, batch 2, joint loss weights 1/1/1, and a checkpoint interval of 25. The engineering preset checkpoints each update. Changing configuration fields marks the preset custom. No data or model weights are silently downloaded by merely selecting a preset.

Joint experiments now also appear in the main experiment registry with the correct ResNet label, analysis counts, Open link and deletion route. They remain accessible from the Joint healthcare page.

## Unchanged scientific scope

This is a selection/integration correction. It does not change the model forward computation, tensor path measures, experimental metric, existing geometry or dataset download source. The publisher download returned HTTP 403 in prior validation; folder and CSV import remain available. The full ResNet metric/BCANw limitations documented in JOINT_TASK_10.md still apply.

## Validation

Regression coverage includes a completely fresh workspace, both entry paths from the main Design page, all three presets, synthetic preparation, creation gating, joint experiment creation and its appearance in the main registry. Python checks cover the catalog before any download and explicit architecture validation. The existing numerical tests are retained.

Executed validation for this package: **87 Python tests passed**. A Chromium test starting with an empty workspace passed both main-page entry paths, preset application, preparation/creation gating, real synthetic preparation, creation and unified registry navigation, with no JavaScript errors. Git patch application passed for both 0.9.0 and 0.8.0, each in project-root and nested repository layouts; resulting file hashes matched the packaged project, workspace data were preserved, and a dirty tree was refused without mutation.
