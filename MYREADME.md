# OCTSB1
```
11/23 03:46:19 - mmengine - INFO - Epoch(train) [1258][100/163]  lr: 1.000000e-05  eta: 0:03:22  time: 0.520031  data_time: 0.012644  memory: 14376  loss: 0.000183  loss/heatmap: 0.000112  loss/displacement: 0.000071
11/23 03:46:23 - mmengine - INFO - Exp name: dekr_testmodel-w32_8xb10-140e_octsegflat-512x512_20231121_182822
11/23 03:46:45 - mmengine - INFO - Epoch(train) [1258][150/163]  lr: 1.000000e-05  eta: 0:02:56  time: 0.521060  data_time: 0.012762  memory: 14376  loss: 0.000178  loss/heatmap: 0.000116  loss/displacement: 0.000062
11/23 03:46:51 - mmengine - INFO - Exp name: dekr_testmodel-w32_8xb10-140e_octsegflat-512x512_20231121_182822
11/23 03:47:17 - mmengine - INFO - Epoch(train) [1259][ 50/163]  lr: 1.000000e-05  eta: 0:02:23  time: 0.523196  data_time: 0.016144  memory: 14376  loss: 0.000186  loss/heatmap: 0.000123  loss/displacement: 0.000063
11/23 03:47:44 - mmengine - INFO - Epoch(train) [1259][100/163]  lr: 1.000000e-05  eta: 0:01:57  time: 0.523867  data_time: 0.016063  memory: 14376  loss: 0.000195  loss/heatmap: 0.000125  loss/displacement: 0.000071
11/23 03:48:10 - mmengine - INFO - Epoch(train) [1259][150/163]  lr: 1.000000e-05  eta: 0:01:31  time: 0.520021  data_time: 0.012742  memory: 14376  loss: 0.000180  loss/heatmap: 0.000116  loss/displacement: 0.000064
11/23 03:48:16 - mmengine - INFO - Exp name: dekr_testmodel-w32_8xb10-140e_octsegflat-512x512_20231121_182822
11/23 03:48:42 - mmengine - INFO - Epoch(train) [1260][ 50/163]  lr: 1.000000e-05  eta: 0:00:58  time: 0.521599  data_time: 0.015162  memory: 14376  loss: 0.000178  loss/heatmap: 0.000114  loss/displacement: 0.000064
11/23 03:49:08 - mmengine - INFO - Epoch(train) [1260][100/163]  lr: 1.000000e-05  eta: 0:00:32  time: 0.521193  data_time: 0.013141  memory: 14376  loss: 0.000211  loss/heatmap: 0.000133  loss/displacement: 0.000078
11/23 03:49:35 - mmengine - INFO - Epoch(train) [1260][150/163]  lr: 1.000000e-05  eta: 0:00:06  time: 0.522587  data_time: 0.013465  memory: 14376  loss: 0.000189  loss/heatmap: 0.000120  loss/displacement: 0.000068

...


Loading and preparing results...
DONE (t=2.05s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=1.41s).
Accumulating evaluation results...
DONE (t=0.22s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.645
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  0.650
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.650
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.933
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.998
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.998
11/23 04:00:28 - mmengine - INFO - Epoch(val) [1260][5715/5715]    coco/AP: 0.645355  coco/AP .5: 0.650285  coco/AP .75: 0.650285  coco/AP (M): -1.000000  coco/AP (L): 0.932906  coco/AR: 0.998463  coco/AR .5: 1.000000  coco/AR .75: 1.000000  coco/AR (M): -1.000000  coco/AR (L): 0.998463  data_time: 0.000598  time: 0.110494
```