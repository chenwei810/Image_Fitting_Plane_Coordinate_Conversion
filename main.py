import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import logging
from PIL import Image  # 添加 PIL Image 導入

class ImageRegistration:
    def __init__(self, reference_band: str = 'Red', method: str = 'SIFT'):
        self.reference_band = reference_band
        self.method = method.lower()
        self.images: Dict[str, np.ndarray] = {}
        self.results: Dict[str, Dict] = {}
        self.errors: Dict[str, float] = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple:
        """特徵檢測和匹配"""
        # 轉換為灰度圖
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 選擇特徵檢測器和匹配器
        if self.method == 'sift':
            detector = cv2.SIFT_create()
            bf = cv2.BFMatcher()
            
            # 檢測特徵點
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                raise ValueError("無法檢測到特徵點")
            
            # 使用 knnMatch 進行特徵匹配
            matches = bf.knnMatch(des1, des2, k=2)
            
            # 應用 Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
        elif self.method == 'surf':
            detector = cv2.xfeatures2d.SURF_create()
            bf = cv2.BFMatcher()
            
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                raise ValueError("無法檢測到特徵點")
            
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
        elif self.method == 'orb':
            detector = cv2.ORB_create(nfeatures=50000)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                raise ValueError("無法檢測到特徵點")
            
            matches = bf.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:100]
            
        else:
            raise ValueError(f"不支援的特徵檢測方法: {self.method}")
        
        return kp1, kp2, good_matches

    def read_images(self, image_dir: str, image_extensions: List[str] = None) -> None:
        """讀取所有影像"""
        if image_extensions is None:
            image_extensions = ['.tif', '.jpg', '.jpeg', '.png', '.bmp']
            
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"目錄不存在: {image_dir}")
            
        image_paths = [f for f in os.listdir(image_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_paths:
            raise ValueError(f"在 {image_dir} 中未找到支援的影像檔案")
        
        for path in tqdm(image_paths, desc="讀取影像"):
            try:
                name = os.path.splitext(path)[0]
                full_path = os.path.join(image_dir, path)
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    self.logger.warning(f"無法讀取影像: {path}")
                    continue
                    
                img = self._preprocess_image(img)
                self.images[name] = img
                self.logger.info(f"成功讀取影像: {path}")
                
            except Exception as e:
                self.logger.error(f"讀取 {path} 時發生錯誤: {str(e)}")

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """影像預處理"""
        if img.dtype == 'uint16':
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype('uint8')
        
        # 對比度增強
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img.shape) == 2:
            img = clahe.apply(img)
        else:
            for i in range(img.shape[2]):
                img[:,:,i] = clahe.apply(img[:,:,i])
            
        # 單通道轉三通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        return img

    def process(self, output_dir: str = 'output') -> None:
        """執行完整的影像套合流程"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.reference_band not in self.images:
            raise ValueError(f"找不到參考波段: {self.reference_band}")
            
        reference_img = self.images[self.reference_band]
        results_for_gif = [reference_img]
        
        for name, img in tqdm(self.images.items(), desc="處理影像"):
            if name == self.reference_band:
                continue
                
            try:
                # 特徵檢測和匹配
                kp1, kp2, matches = self.detect_and_match_features(reference_img, img)
                
                # 配準和分析
                results = self._process_single_image(
                    reference_img, img, kp1, kp2, matches)
                
                if results is None:
                    continue
                    
                warped_img, matches_viz, error_viz, rmse = results
                results_for_gif.append(warped_img)
                
                # 保存結果
                self._save_results(name, warped_img, matches_viz, error_viz, 
                                 rmse, output_dir)
                
            except Exception as e:
                self.logger.error(f"處理 {name} 時發生錯誤: {str(e)}")
                continue

        # 創建GIF動畫
        if results_for_gif:
            self._create_gif(results_for_gif, os.path.join(output_dir, 'registration_result.gif'))

    def _process_single_image(self, reference_img: np.ndarray, 
                            img: np.ndarray, kp1, kp2, matches) -> Optional[Tuple]:
        """處理單張影像的配準"""
        if len(matches) < 4:
            self.logger.warning("匹配點太少，無法進行配準")
            return None
            
        # 提取匹配點
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 估算變換矩陣
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            self.logger.warning("無法找到合適的變換矩陣")
            return None
            
        # 影像變換
        warped = cv2.warpPerspective(img, M, (reference_img.shape[1], 
                                             reference_img.shape[0]))
        
        # 創建視覺化結果
        matches_viz = self._create_matches_visualization(
            reference_img, img, kp1, kp2, [m for i, m in enumerate(matches) if mask[i]])
        error_viz = self._create_error_visualization(reference_img, warped)
        
        # 計算誤差
        rmse = self._calculate_rmse(reference_img, warped)
        
        return warped, matches_viz, error_viz, rmse

    def _create_matches_visualization(self, img1: np.ndarray, img2: np.ndarray, 
                                    kp1, kp2, matches) -> np.ndarray:
        """創建特徵點匹配的視覺化結果"""
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def _create_error_visualization(self, img1: np.ndarray, 
                                  img2: np.ndarray) -> np.ndarray:
        """創建誤差視覺化"""
        diff = cv2.absdiff(img1, img2)
        normalized_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        colored_diff = cv2.applyColorMap(normalized_diff.astype(np.uint8), 
                                       cv2.COLORMAP_JET)
        return cv2.addWeighted(img1, 0.3, colored_diff, 0.7, 0)

    def _create_gif(self, images: List[np.ndarray], output_path: str, 
                   duration: int = 500) -> None:
        """創建GIF動畫"""
        try:
            pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
                         for img in images]
            pil_images[0].save(
                output_path, 
                save_all=True, 
                append_images=pil_images[1:],
                duration=duration, 
                loop=0
            )
            self.logger.info(f"已成功創建GIF動畫：{output_path}")
        except Exception as e:
            self.logger.error(f"創建GIF動畫時發生錯誤：{str(e)}")

    def _calculate_rmse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """計算均方根誤差"""
        return np.sqrt(np.mean((img1.astype(float) - img2.astype(float)) ** 2))

    def _save_results(self, name: str, warped_img: np.ndarray, 
                     matches_viz: np.ndarray, error_viz: np.ndarray, 
                     rmse: float, output_dir: str) -> None:
        """保存處理結果"""
        # 保存到記憶體
        self.results[name] = {
            'warped': warped_img,
            'matches_viz': matches_viz,
            'error_viz': error_viz
        }
        self.errors[name] = rmse
        
        # 儲存檔案
        cv2.imwrite(os.path.join(output_dir, f'{name}_warped.tif'), warped_img)
        cv2.imwrite(os.path.join(output_dir, f'{name}_matches.png'), matches_viz)
        cv2.imwrite(os.path.join(output_dir, f'{name}_error.png'), error_viz)
        
        # 記錄誤差
        with open(os.path.join(output_dir, 'registration_errors.txt'), 'a') as f:
            f.write(f"{name}: RMSE = {rmse:.2f}\n")

def main():
    try:
        # 設置參數
        input_dir = 'spectral_images'
        output_dir = 'output'
        reference_band = 'Green'  # 設置參考波段
        method = 'SIFT'  # 可選 'SIFT', 'SURF', 或 'ORB'
        
        # 建立影像配準物件
        registrator = ImageRegistration(reference_band=reference_band, method=method)
        
        # 讀取影像
        registrator.read_images(input_dir)
        
        # 執行配準流程
        registrator.process(output_dir)
        
    except Exception as e:
        logging.error(f"程式執行過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()