# 遙測影像多波段套合報告

## 摘要
本研究實現了一個自動化的多波段遙測影像套合系統，使用SIFT特徵提取和仿射變換進行影像對位。系統能夠處理不同波段（Blue、Green、NIR、Red、RedEdge）的TIF格式影像，並提供詳細的視覺化分析和誤差評估。

## 1. 研究背景與目的

### 1.1 研究背景
多光譜遙測影像在獲取過程中，不同波段可能存在位移和變形，需要精確的影像套合來確保後續分析的準確性。

### 1.2 研究目的
- 實現多波段遙測影像的自動化套合
- 提供套合質量的定量評估
- 生成視覺化結果用於套合效果分析

## 2. 方法與實現

### 2.1 系統架構
程式採用模組化設計，主要包含以下功能：
1. 影像讀取與預處理
2. 特徵點提取與匹配
3. 幾何轉換估算與實現
4. 套合效果評估與視覺化

### 2.2 技術細節

#### 2.2.1 影像預處理
- 支援16位元TIF格式讀取
- 自動進行位深轉換（16位元轉8位元）
- 單波段轉RGB處理

#### 2.2.2 特徵提取與匹配
- **特徵提取：** 使用SIFT算法
  - 尺度不變性
  - 旋轉不變性
  - 光照變化適應性
- **特徵匹配：** 
  - 使用BruteForce匹配器
  - 採用交叉檢驗提高可靠性
  - 基於距離比率進行匹配點篩選

#### 2.2.3 幾何轉換
- **轉換模型：** 仿射變換（Affine Transformation）
  - 6個自由度
  - 保持平行關係
  - 適合航拍影像校正
- **RANSAC算法：**
  - 剔除異常匹配點
  - 提高模型穩定性

### 2.3 實現流程
1. 影像讀取與格式轉換
2. 特徵點提取與匹配
3. 轉換模型估算
4. 影像變換與重採樣
5. 結果評估與視覺化

## 3. 實驗結果

### 3.1 套合精度分析
- 均方根誤差（RMSE）統計
- 各波段誤差分布分析
- 特徵點分布評估

### 3.2 視覺化結果
1. **特徵點匹配圖：**
   - 展示匹配點對
   - 反映特徵分布
   
2. **誤差熱圖：**
   - 顯示局部變形程度
   - 識別問題區域

3. **動態效果：**
   - GIF動畫展示各波段套合效果
   - 直觀評估套合質量

## 4. 系統使用說明

### 4.1 環境需求
```bash
pip install opencv-python numpy matplotlib imageio tqdm tifffile
```

### 4.2 使用方法
1. 準備影像文件：
   - 支援格式：TIF
   - 命名要求：Blue.tif, Green.tif, NIR.tif, Red.tif, RedEdge.tif

2. 執行程序：
   ```python
   python image_registration.py
   ```

3. 輸出結果：
   - 套合後影像（*_warped.tif）
   - 匹配結果（*_matches.tif）
   - 誤差視覺化（*_error.tif）
   - 動態展示（registration_result.gif）

## 5. 結論與展望

### 5.1 主要成果
- 實現了多波段影像的自動化套合
- 提供了完整的視覺化分析工具
- 達到了較高的套合精度

### 5.2 改進方向
1. **算法優化：**
   - 考慮添加其他特徵提取方法（如ORB）
   - 優化匹配策略

2. **功能擴展：**
   - 支援批量處理
   - 添加交互式參數調整
   - 擴展支援更多影像格式

3. **效能提升：**
   - 優化大尺寸影像處理
   - 提高運算