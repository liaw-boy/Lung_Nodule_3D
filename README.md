# 3D Lung Nodule Classification

3D CNN 肺結節良惡性分類。

延續上游 [LIDC_IDRI](https://github.com/liaw-boy/LIDC_IDRI) 2D 系統，改用 3D ResNet 處理結節體素，目標是提升微小結節 (<6mm) 的判讀敏感度，以及加入 LUNA16 多中心資料驗證跨資料集泛化能力。

---

## 目標

1. 微小結節 (<6mm) sensitivity 比 2D + 3D 聚合提升 ≥ 5%
2. 在 LUNA16 跨資料集達 ≥ 95% sensitivity
3. 模型大小 < 50MB，inference < 200ms

---

## 架構

```
[DICOM volume / stacked PNG]
          ↓
[Volume Pre-processor]   取 64×64×16 voxel ROI
          ↓
[3D ResNet18 + CBAM]     3D Conv 卷積 + 注意力
          ↓
[Aux Head] + [Malignancy Head]   AttFeedback
          ↓
[Lung-RADS 分級]
```

---

## 目錄結構

```
Lung_Nodule_3D/
├── data/
│   ├── build_3d_volumes.py      PNG → 3D volume
│   └── lidc_3d_dataset.py       PyTorch Dataset
├── models/
│   ├── resnet3d.py              3D ResNet18
│   ├── cbam3d.py                3D CBAM
│   └── nodule_classifier_3d.py  分類器
└── scripts/
    └── train_3d.py              訓練主程式
```

---

## 階段規劃

### 階段 A — Pseudo-3D（先跑驗證架構）
資料來源：上游 PNG 切片，插值到固定 16 voxel Z 軸。
驗證：3D ResNet18 是否比 2D AttFB 在同一 holdout 拿更高 AUC。
Risk：Z 軸是插值不是真物理距離。

### 階段 B — DICOM 3D（看 A 結果決定）
重抓 LIDC raw DICOM 拿真實 voxel spacing。
加 LUNA16 多中心資料，訓練 3D ResNet50 / DenseNet121。

### 階段 C — 部署整合
把 3D model 封裝進原 GUI 作為 v3 升級。
保留 2D 為 fallback（3D 推理較慢）。

---

## 與上游專案對照

| 項目 | LIDC_IDRI (2D) | Lung_Nodule_3D |
|:---|:---|:---|
| 模型 | YOLO11n + 2D CNN + 3D Gaussian | 3D ResNet + CBAM |
| 資料表示 | PNG slice (per-slice) | 3D voxel ROI (per-nodule) |
| 召回 baseline | 100% (33/33) | 待驗證 |
| 推理速度 | ~0.5s / nodule | 預估 ~2-5s / nodule |
| Patient split | seed=42 (51 test) | 同一 split（公平比較） |

---

## 資料合規

沿用上游 patient-level split (seed=42)，同一 51 病患 holdout test set，確保和 2D baseline 比較是 apples-to-apples。

---

## 當前狀態 (Sprint 0)

- [x] Repo init + 結構規劃
- [ ] PNG → 3D volume 重建腳本
- [ ] 3D ResNet18 baseline
- [ ] 階段 A 第一次訓練 + KPI vs 2D 對照

---

## 研究團隊

- **指導老師**：淡江大學資訊工程系 教授
- **組員**：陳威丞、廖柏維、鍾翔宇、江昊宸

---

*Future work: LUNA16 multi-center validation, model compression for edge deployment.*
