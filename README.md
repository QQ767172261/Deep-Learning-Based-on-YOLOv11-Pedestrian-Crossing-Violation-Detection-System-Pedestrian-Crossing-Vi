基于 YOLOv11 行人过马路违规检测系统  行人过马路违规检测数据集，1967张（存在增强处理，图像通过视频抽帧抽取，有一定相似），yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e6cf9adcd3545f5a556e7878e937a1d.png)

图像尺寸:416*416
类别数量:2类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2f0c8aa0ca3e4775a586cfb2d96f6184.png)

训练集图像数量:1816; 验证集图像数量:90； 测试集图像数量:61
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ee6cc0e45904abb872a0c15af8c77b8.png)

Crosswalk_Violation 走斑马线过马路: 1068,1068
Crossing_Violation 不按规定横穿马路: 899,899
image num: 1967![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14b33f468bec4ef594f3c1eb41d25741.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13e25101e1464d2fa3bd3ab84b599c6b.png)



模型代码：提供全部训练及测试源代码，模型训练使用yolov11n训练，50个epoch训练结果，map如描述图所示。


qt界面：运行界面采用pyqt5编写， 提供全部源代码，支持图片、视频及摄像头进行检测: 界面可实时显示目标位置、目标总数、置信度等信息: 支持检测结果保存;本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像


# 基于YOLOv11行人过马路违规检测系统完整代码
## 1、数据集yaml配置 `cross_violation.yaml`
```yaml
path: ./crosswalk_dataset
train: images/train
val: images/val
test: images/test

names:
  0: Crosswalk_Violation
  1: Crossing_Violation
```
>中文标签：
Crosswalk_Violation：人行横道违规；Crossing_Violation：横穿马路违规

## 2、模型训练代码 train.py
```python
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv11n模型
    model = YOLO("yolov11n.pt")
    results = model.train(
        data="cross_violation.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="crosswalk_result",
        name="yolov11_cross_violation",
        save=True,
        val=True
    )
    print("训练完成，最佳权重：crosswalk_result/yolov11_cross_violation/weights/best.pt")
```

## 3、推理代码 predict.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("crosswalk_result/yolov11_cross_violation/weights/best.pt")

def detect_image(img_path, save_path="cross_result.jpg"):
    res = model.predict(img_path, conf=0.4)
    draw_img = res[0].plot()
    cv2.imwrite(save_path, draw_img)
    return draw_img

def detect_video(video_path):
    model.predict(source=video_path, save=True, conf=0.4)

if __name__ == "__main__":
    detect_image("test.jpg")
```

## 4、PyQt5可视化界面代码
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QComboBox, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO

class CrossDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的行人过马路违规检测系统")
        self.setGeometry(50, 50, 1280, 900)
        self.model = YOLO("crosswalk_result/yolov11_cross_violation/weights/best.pt")
        self.result_img = None
        self.img_path = ""

        # 图片显示区域
        self.img_label = QLabel(self)
        self.img_label.setGeometry(30, 80, 720, 520)
        self.img_label.setStyleSheet("border:1px solid #333;")

        # 右侧控件
        label_file = QLabel("文件导入", self)
        label_file.setGeometry(780,80,120,30)

        self.btn_img = QPushButton("选择图片", self)
        self.btn_img.setGeometry(780,110,180,35)
        self.btn_img.clicked.connect(self.load_image)

        self.btn_video = QPushButton("选择视频文件", self)
        self.btn_video.setGeometry(780,150,180,35)

        self.btn_cam = QPushButton("开启摄像头", self)
        self.btn_cam.setGeometry(780,190,180,35)

        label_conf = QLabel("置信度", self)
        label_conf.setGeometry(780,240,80,30)
        self.combo_conf = QComboBox(self)
        self.combo_conf.addItems(["0.3","0.4","0.5","0.6","0.7"])
        self.combo_conf.setGeometry(830,240,80,30)

        label_target = QLabel("目标选择", self)
        label_target.setGeometry(780,280,80,30)
        self.combo_target = QComboBox(self)
        self.combo_target.addItems(["全部"])
        self.combo_target.setGeometry(830,280,80,30)

        label_pos = QLabel("目标位置", self)
        label_pos.setGeometry(780,330,120,30)
        self.text_pos = QTextEdit(self)
        self.text_pos.setGeometry(780,360,220,120)

        self.btn_save = QPushButton("保存", self)
        self.btn_save.setGeometry(780,500,100,40)
        self.btn_save.clicked.connect(self.save_result)

        self.btn_exit = QPushButton("退出", self)
        self.btn_exit.setGeometry(900,500,100,40)
        self.btn_exit.clicked.connect(self.close)

        # 底部结果表格
        self.table = QTableWidget(self)
        self.table.setGeometry(30,630,1020,180)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "Image Files(*.jpg *.png *.jpeg)")
        if not fname:
            return
        self.img_path = fname
        conf_thresh = float(self.combo_conf.currentText())
        results = self.model.predict(fname, conf=conf_thresh)
        self.table.setRowCount(0)
        for r in results:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row,0,QTableWidgetItem(str(row+1)))
                self.table.setItem(row,1,QTableWidgetItem(fname))
                self.table.setItem(row,2,QTableWidgetItem(cls_name))
                self.table.setItem(row,3,QTableWidgetItem(f"{conf:.2f}"))
                coord_str = f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
                self.table.setItem(row,4,QTableWidgetItem(coord_str))
                self.text_pos.setText(f"xmin:{int(x1)}\nymin:{int(y1)}\nxmax:{int(x2)}\nymax:{int(y2)}")
        img_draw = results[0].plot()
        self.result_img = img_draw
        img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, c*w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(), Qt.KeepAspectRatio))

    def save_result(self):
        if self.result_img is not None:
            cv2.imwrite("cross_detect_save.jpg", self.result_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CrossDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

## 5、PR曲线绘图代码
```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("crosswalk_result/yolov11_cross_violation/weights/best.pt")
res = model.val()
pr_data = res.plots["PR_curve"]
plt.figure(figsize=(10,7))
plt.plot(pr_data[0],pr_data[1],label=f"all classes {res.box.map50:.3f} mAP@0.5",linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("cross_pr_curve.png",dpi=300)
plt.show()
```

## 环境安装
```bash
pip install ultralytics opencv-python pyqt5
```

## 应用场景
1. 城市道路监控摄像头识别行人、非机动车违规横穿马路、斑马线违规行为
2. 智慧交通，自动抓拍道路行人违章，辅助交管预警
3. 毕业设计、目标检测算法研究、交通违规智能识别项目

