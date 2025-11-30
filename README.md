# Real-ESRGAN: 高効率・深度分離ハイブリッド計画

このリポジトリでは、Real-ESRGANに匹敵する画質を保ちつつ、計算コストとVRAM消費を大幅に削減する**高効率・深度分離ハイブリッド**モデルを実装・検証します。

## アーキテクチャ概要

### ネットワーク構成
| 部位 | モジュール | 機能・特徴 | パラメータ (推奨) | コスト削減貢献度 |
| :--- | :--- | :--- | :--- | :--- |
| 浅い特徴抽出 | `nn.Conv2d` | LR画像から初期特徴 $\mathbf{F}_0$ を抽出。 | $3 \times 3$, **NF=32** | **高:** NF削減で全体コスト低減 |
| 深い特徴抽出 | `ERBlock` × 8 | 主な復元処理（全体残差接続内）。 | **8層** | **高:** ブロック数削減 |
| トランクConv | `nn.Conv2d` | 深い特徴 $\mathbf{F}_{deep}$ を整形。 | $3 \times 3$, NF=32 | - |
| アップサンプリング | `Conv → PixelShuffle → Conv → PixelShuffle` | 2段の`PixelShuffle(r=2)`で×4拡大。 | 2段のPS | **中:** 転置畳み込みより効率的 |
| 最終出力 | `nn.Conv2d` | HR画像を出力。 | $3 \times 3$, 3ch | - |

### ERBlock 詳細
| 部位 | モジュール | 詳細設計 | 計算コストへの影響 |
| :--- | :--- | :--- | :--- |
| 空間フィルタ | Depthwise Conv | カーネル $3 \times 3$, groups=NF | **極めて低い**（通常Convの約1/9） |
| チャネル結合 | Pointwise Conv | カーネル $1 \times 1$ | 空間演算なしで効率的 |
| チャネル注意 | ECA-Lite | `AvgPool → 1x1 Conv → Sigmoid` | **非常に低い** |
| 活性化 | `nn.ReLU` | Depthwise後に配置 | - |
| 最終出力 | $\mathbf{x} + \mathbf{Residual}$ | 入力へ残差を加算 | - |

## 目標
- **FLOPs:** 約70〜80%削減（チャネル削減＋深度分離畳み込み）
- **推論速度:** 2〜3倍高速化（量子化併用でさらに高速化）
- **画質:** Real-ESRGAN標準モデルと同等以上の実写復元性能

## 実装フェーズと進捗
- [x] **フェーズ1:** ERBlock単体実装と機能テスト
- [x] **フェーズ2:** EfficientSRNet全体構造実装と形状テスト
- [x] フェーズ3: 損失関数とデータローダー設定
- [ ] フェーズ4: 学習実行、速度・メモリベンチマーク
- [ ] フェーズ5: 画質評価と結果比較

### フェーズ3: 損失関数とデータローダー
- **データローダー:** `SuperResolutionDataset` が HR/LR ペアを読み込み、LRが無い場合は HR からバイキュービックで生成します。ランダムクロップ、左右/上下フリップの軽量オーグメントをサポートし、`build_dataloaders` で訓練・検証ローダーを一括構築できます。
- **損失:** `CompositeLoss` は L1、Charbonnier、Total Variation の重み付き合成をサポートします。`build_loss` から簡単に作成でき、個別の損失項も辞書で返してログに使えます。
- **使用例:**
```python
from pathlib import Path
from training_utils import DataloaderConfig, LossConfig, build_dataloaders, build_loss

train_conf = DataloaderConfig(hr_dir=Path('data/train_HR'), lr_dir=Path('data/train_LR'), batch_size=8, scale=4)
loaders = build_dataloaders(train_conf)

loss_fn = build_loss(LossConfig(pixel_weight=1.0, charbonnier_weight=0.5, tv_weight=0.1))
total_loss, components = loss_fn(sr_pred, hr_gt)
```

## ローカルテスト
### 実装済みチェック項目
- 形状チェック: $(1, 32, 64, 64) \rightarrow (1, 32, 64, 64)$ (ERBlock) / $(1, 3, H, W) \rightarrow (1, 3, 4H, 4W)$ (EfficientSRNet)
- 残差接続チェック: 出力と入力の差分に非ゼロ要素が存在
- Attention動作: Sigmoid出力が0〜1に収まり、分散が0でないこと
- パラメータ予算: 2M未満に抑制（Real-ESRGAN標準より大幅に少ない想定）
- NaN/Inf無しの順伝播確認

### 実行方法
1. 依存関係（PyTorchなど）をインストールします。
2. ルートで以下を実行して合成テストデータによる形状・注意・パラメータの検証を行います。

```bash
pytest -q
```

`tests/test_efficient_sr.py` では、ランダムテンソルを用いた以下の検証を行います。
- ERBlockの形状と残差の非ゼロ性
- ECA-Liteのチャネル注意が0〜1に収まり、チャネル間で変動すること
- EfficientSRNetの×4/×2出力形状とNaN非発生
- パラメータ総数が閾値未満であること

フェーズ3以降ではデータローダー、学習スクリプト、速度・画質評価の結果をこのREADMEに追記して進捗を更新していきます。
