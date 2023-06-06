# AreaDetection

## 線上執行
[Colab 連結](https://drive.google.com/file/d/1pwpwaMuDQ1lEdVII3iQcA-TpFtHMkBXT/view?usp=sharing)

測試結果 (無 UI 介面，僅能推論圖像,影片)

![image](https://github.com/ruoming1021/AreaDetection/assets/43459716/a234e427-5661-47c1-92e8-045167529de9)

## 本地執行

由於我有修改過 YOLOv8 作者的程式碼，所以需要下載
[ultralytics資料夾](https://drive.google.com/file/d/1NlDIarGYVTp4itnej2pBuvzVa9ZGe8TZ/view?usp=drive_link)
，如果用作者的會執行不了

安裝 ultralytics 資料夾裡的 requirements.txt 以及 ultralytics 套件，環境需要 [**Python>=3.7**](https://www.python.org/), [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).
```bash
pip install -r requirements.txt
pip install ultralytics
```
執行程式
```bash
python3 my_couse.py
```
測試結果：https://drive.google.com/file/d/1z0CVy1gxKv9XCzoruHwE5n6Tg0FGB1er/view?usp=share_link
