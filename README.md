
01/08 08:55:37 - mmengine - WARNING - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
01/08 08:55:37 - mmengine - WARNING - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
01/08 08:55:37 - mmengine - INFO - Checkpoints will be saved to /home/user/projects/mmaction2-main/work_dirs/slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.
01/08 08:55:39 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:55:39 - mmengine - INFO - Epoch(train)  [1][4/4]  lr: 1.0000e-02  eta: 0:00:46  time: 0.6172  data_time: 0.1568  memory: 18417  grad_norm: 10.1857  loss: 1.1313  recall@thr=0.5: 0.3889  prec@thr=0.5: 0.3889  recall@top3: 0.7222  prec@top3: 0.2407  recall@top5: 0.9444  prec@top5: 0.1889  loss_action_cls: 1.1313
01/08 08:55:39 - mmengine - INFO - Saving checkpoint at 1 epochs
==> 0.000263929 seconds to Reading GT results
==> 0.00018239 seconds to Reading Detection results
==> 0.840085 seconds to Calculating TP/FP
==> 0.00151896 seconds to Run Evaluator
Per-class results:
Index: 1, Action: Fetch: AP: nan;
Index: 2, Action: Power-on-Test: AP: 0.0000;
Index: 3, Action: Scan: AP: 0.0000;
Index: 4, Action: Labeling: AP: 0.0000;
Index: 5, Action: Visual-Inspection: AP: nan;
Index: 6, Action: working: AP: 0.0000;
Index: 7, Action: Waiting: AP: 0.0000;
Index: 8, Action: Interrupted: AP: nan;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Fetch AP: nan
Class Power-on-Test AP: 0.0000
Class Scan AP: 0.0000
Class Labeling AP: 0.0000
Class Visual-Inspection AP: nan
Class working AP: 0.0000
Class Waiting AP: 0.0000
Class Interrupted AP: nan
01/08 08:55:43 - mmengine - INFO - Epoch(val) [1][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0382  time: 0.0654
01/08 08:55:45 - mmengine - INFO - The best checkpoint with 0.0000 mAP/overall at 1 epoch is saved to best_mAP_overall_epoch_1.pth.
01/08 08:55:49 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:55:49 - mmengine - INFO - Epoch(train)  [2][4/4]  lr: 3.2500e-02  eta: 0:00:36  time: 0.5114  data_time: 0.1321  memory: 18417  grad_norm: 9.0480  loss: 1.4848  recall@thr=0.5: 0.5882  prec@thr=0.5: 0.5882  recall@top3: 0.6765  prec@top3: 0.2353  recall@top5: 0.9118  prec@top5: 0.1882  loss_action_cls: 1.4848
01/08 08:55:49 - mmengine - INFO - Saving checkpoint at 2 epochs
==> 0.000188351 seconds to Reading GT results
==> 9.29832e-05 seconds to Reading Detection results
==> 0.91178 seconds to Calculating TP/FP
==> 0.000650406 seconds to Run Evaluator
Per-class results:
Index: 8, Action: Interrupted: AP: nan;
Index: 3, Action: Scan: AP: 0.0000;
Index: 5, Action: Visual-Inspection: AP: nan;
Index: 6, Action: working: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Interrupted AP: nan
Class Scan AP: 0.0000
Class Visual-Inspection AP: nan
Class working AP: 0.0000
01/08 08:55:52 - mmengine - INFO - Epoch(val) [2][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0084  time: 0.0252
01/08 08:55:54 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:55:54 - mmengine - INFO - Epoch(train)  [3][4/4]  lr: 5.5000e-02  eta: 0:00:32  time: 0.4764  data_time: 0.1219  memory: 18417  grad_norm: 9.1577  loss: 1.7206  recall@thr=0.5: 0.4706  prec@thr=0.5: 0.2353  recall@top3: 0.7647  prec@top3: 0.2549  recall@top5: 0.9412  prec@top5: 0.1882  loss_action_cls: 1.7206
01/08 08:55:54 - mmengine - INFO - Saving checkpoint at 3 epochs
==> 0.000202656 seconds to Reading GT results
==> 4.91142e-05 seconds to Reading Detection results
==> 0.861462 seconds to Calculating TP/FP
==> 0.000538111 seconds to Run Evaluator
Per-class results:
Index: 6, Action: working: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class working AP: 0.0000
01/08 08:55:57 - mmengine - INFO - Epoch(val) [3][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0088  time: 0.0253
01/08 08:55:59 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:55:59 - mmengine - INFO - Epoch(train)  [4][4/4]  lr: 7.7500e-02  eta: 0:00:29  time: 0.4583  data_time: 0.1186  memory: 18417  grad_norm: 8.7057  loss: 2.0640  recall@thr=0.5: 0.3333  prec@thr=0.5: 0.3333  recall@top3: 0.8333  prec@top3: 0.2778  recall@top5: 0.8889  prec@top5: 0.1778  loss_action_cls: 2.0640
01/08 08:55:59 - mmengine - INFO - Saving checkpoint at 4 epochs
==> 0.000167608 seconds to Reading GT results
==> 7.05719e-05 seconds to Reading Detection results
==> 0.925533 seconds to Calculating TP/FP
==> 0.000557184 seconds to Run Evaluator
Per-class results:
Index: 5, Action: Visual-Inspection: AP: nan;
Index: 6, Action: working: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Visual-Inspection AP: nan
Class working AP: 0.0000
01/08 08:56:02 - mmengine - INFO - Epoch(val) [4][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0083  time: 0.0246
01/08 08:56:04 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:04 - mmengine - INFO - Epoch(train)  [5][4/4]  lr: 1.0000e-01  eta: 0:00:26  time: 0.4497  data_time: 0.1159  memory: 18417  grad_norm: 7.9418  loss: 2.6365  recall@thr=0.5: 0.4706  prec@thr=0.5: 0.4706  recall@top3: 0.5882  prec@top3: 0.1961  recall@top5: 0.8235  prec@top5: 0.1647  loss_action_cls: 2.6365
01/08 08:56:04 - mmengine - INFO - Saving checkpoint at 5 epochs
==> 0.000170469 seconds to Reading GT results
==> 4.95911e-05 seconds to Reading Detection results
==> 0.887403 seconds to Calculating TP/FP
==> 0.000225306 seconds to Run Evaluator
Per-class results:
Index: 5, Action: Visual-Inspection: AP: nan;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:279: RuntimeWarning: Mean of empty slice
  overall = np.nanmean([x[2] for x in cls_AP])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:280: RuntimeWarning: Mean of empty slice
  person_movement = np.nanmean([x[2] for x in cls_AP if x[0] <= 14])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: nan
Person Movement mAP: nan
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Visual-Inspection AP: nan
01/08 08:56:07 - mmengine - INFO - Epoch(val) [5][8/8]    mAP/overall: nan  mAP/person_movement: nan  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0086  time: 0.0260
01/08 08:56:09 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:09 - mmengine - INFO - Epoch(train)  [6][4/4]  lr: 1.0000e-01  eta: 0:00:24  time: 0.4060  data_time: 0.1051  memory: 18417  grad_norm: 6.5662  loss: 3.0758  recall@thr=0.5: 0.3125  prec@thr=0.5: 0.1875  recall@top3: 0.6250  prec@top3: 0.2083  recall@top5: 0.8750  prec@top5: 0.1750  loss_action_cls: 3.0758
01/08 08:56:09 - mmengine - INFO - Saving checkpoint at 6 epochs
==> 0.000165701 seconds to Reading GT results
==> 6.8903e-05 seconds to Reading Detection results
==> 0.932647 seconds to Calculating TP/FP
==> 0.000595093 seconds to Run Evaluator
Per-class results:
Index: 6, Action: working: AP: 0.0000;
Index: 7, Action: Waiting: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class working AP: 0.0000
Class Waiting AP: 0.0000
01/08 08:56:12 - mmengine - INFO - Epoch(val) [6][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0084  time: 0.0259
01/08 08:56:13 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:13 - mmengine - INFO - Epoch(train)  [7][4/4]  lr: 1.0000e-01  eta: 0:00:22  time: 0.4063  data_time: 0.1059  memory: 18417  grad_norm: 5.6563  loss: 3.0837  recall@thr=0.5: 0.5556  prec@thr=0.5: 0.4444  recall@top3: 0.7222  prec@top3: 0.2407  recall@top5: 0.8333  prec@top5: 0.1667  loss_action_cls: 3.0837
01/08 08:56:13 - mmengine - INFO - Saving checkpoint at 7 epochs
==> 0.000202179 seconds to Reading GT results
==> 7.27177e-05 seconds to Reading Detection results
==> 0.896561 seconds to Calculating TP/FP
==> 0.000526667 seconds to Run Evaluator
Per-class results:
Index: 2, Action: Power-on-Test: AP: 0.0000;
Index: 5, Action: Visual-Inspection: AP: nan;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Power-on-Test AP: 0.0000
Class Visual-Inspection AP: nan
01/08 08:56:17 - mmengine - INFO - Epoch(val) [7][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0089  time: 0.0249
01/08 08:56:18 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:18 - mmengine - INFO - Epoch(train)  [8][4/4]  lr: 1.0000e-01  eta: 0:00:20  time: 0.4075  data_time: 0.1070  memory: 18417  grad_norm: 4.3364  loss: 2.9248  recall@thr=0.5: 0.3529  prec@thr=0.5: 0.3529  recall@top3: 0.8235  prec@top3: 0.2745  recall@top5: 1.0000  prec@top5: 0.2000  loss_action_cls: 2.9248
01/08 08:56:18 - mmengine - INFO - Saving checkpoint at 8 epochs
==> 0.000171185 seconds to Reading GT results
==> 5.81741e-05 seconds to Reading Detection results
==> 0.899605 seconds to Calculating TP/FP
==> 0.000226498 seconds to Run Evaluator
Per-class results:
Index: 5, Action: Visual-Inspection: AP: nan;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:279: RuntimeWarning: Mean of empty slice
  overall = np.nanmean([x[2] for x in cls_AP])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:280: RuntimeWarning: Mean of empty slice
  person_movement = np.nanmean([x[2] for x in cls_AP if x[0] <= 14])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: nan
Person Movement mAP: nan
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Visual-Inspection AP: nan
01/08 08:56:22 - mmengine - INFO - Epoch(val) [8][8/8]    mAP/overall: nan  mAP/person_movement: nan  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0078  time: 0.0248
01/08 08:56:23 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:23 - mmengine - INFO - Epoch(train)  [9][4/4]  lr: 1.0000e-01  eta: 0:00:18  time: 0.4088  data_time: 0.1067  memory: 18417  grad_norm: 3.1376  loss: 2.4681  recall@thr=0.5: 0.2941  prec@thr=0.5: 0.2941  recall@top3: 0.7647  prec@top3: 0.2549  recall@top5: 0.9412  prec@top5: 0.1882  loss_action_cls: 2.4681
01/08 08:56:23 - mmengine - INFO - Saving checkpoint at 9 epochs
==> 0.000163078 seconds to Reading GT results
==> 6.07967e-05 seconds to Reading Detection results
==> 0.917267 seconds to Calculating TP/FP
==> 0.000611544 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
Index: 6, Action: working: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
Class working AP: 0.0000
01/08 08:56:27 - mmengine - INFO - Epoch(val) [9][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0077  time: 0.0237
01/08 08:56:28 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:28 - mmengine - INFO - Epoch(train) [10][4/4]  lr: 1.0000e-01  eta: 0:00:17  time: 0.4086  data_time: 0.1072  memory: 18417  grad_norm: 2.3949  loss: 1.5954  recall@thr=0.5: 0.1250  prec@thr=0.5: 0.0937  recall@top3: 0.7500  prec@top3: 0.2500  recall@top5: 1.0000  prec@top5: 0.2000  loss_action_cls: 1.5954
01/08 08:56:28 - mmengine - INFO - Saving checkpoint at 10 epochs
==> 0.00017786 seconds to Reading GT results
==> 4.98295e-05 seconds to Reading Detection results
==> 0.899354 seconds to Calculating TP/FP
==> 0.000514984 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
01/08 08:56:31 - mmengine - INFO - Epoch(val) [10][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0085  time: 0.0250
01/08 08:56:33 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:33 - mmengine - INFO - Epoch(train) [11][4/4]  lr: 1.0000e-02  eta: 0:00:15  time: 0.4098  data_time: 0.1062  memory: 18417  grad_norm: 2.5687  loss: 1.0754  recall@thr=0.5: 0.3824  prec@thr=0.5: 0.3824  recall@top3: 0.7647  prec@top3: 0.2745  recall@top5: 0.8824  prec@top5: 0.1882  loss_action_cls: 1.0754
01/08 08:56:33 - mmengine - INFO - Saving checkpoint at 11 epochs
==> 0.000186682 seconds to Reading GT results
==> 4.91142e-05 seconds to Reading Detection results
==> 0.889019 seconds to Calculating TP/FP
==> 0.000509977 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
01/08 08:56:37 - mmengine - INFO - Epoch(val) [11][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0084  time: 0.0257
01/08 08:56:38 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:38 - mmengine - INFO - Epoch(train) [12][4/4]  lr: 1.0000e-02  eta: 0:00:13  time: 0.4103  data_time: 0.1046  memory: 18417  grad_norm: 2.2977  loss: 0.8543  recall@thr=0.5: 0.3333  prec@thr=0.5: 0.3333  recall@top3: 0.8333  prec@top3: 0.2778  recall@top5: 0.8889  prec@top5: 0.1778  loss_action_cls: 0.8543
01/08 08:56:38 - mmengine - INFO - Saving checkpoint at 12 epochs
==> 0.000159979 seconds to Reading GT results
==> 5.126e-05 seconds to Reading Detection results
==> 0.909508 seconds to Calculating TP/FP
==> 0.000497341 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
01/08 08:56:41 - mmengine - INFO - Epoch(val) [12][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0084  time: 0.0263
01/08 08:56:43 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:43 - mmengine - INFO - Epoch(train) [13][4/4]  lr: 1.0000e-02  eta: 0:00:11  time: 0.4082  data_time: 0.1034  memory: 18417  grad_norm: 1.9316  loss: 0.6718  recall@thr=0.5: 0.2667  prec@thr=0.5: 0.2667  recall@top3: 0.8000  prec@top3: 0.2667  recall@top5: 0.8333  prec@top5: 0.1733  loss_action_cls: 0.6718
01/08 08:56:43 - mmengine - INFO - Saving checkpoint at 13 epochs
==> 0.000175714 seconds to Reading GT results
==> 4.55379e-05 seconds to Reading Detection results
==> 0.918091 seconds to Calculating TP/FP
==> 0.000524521 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
01/08 08:56:46 - mmengine - INFO - Epoch(val) [13][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0088  time: 0.0254
01/08 08:56:48 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:48 - mmengine - INFO - Epoch(train) [14][4/4]  lr: 1.0000e-02  eta: 0:00:10  time: 0.4077  data_time: 0.1026  memory: 18417  grad_norm: 1.8908  loss: 0.6166  recall@thr=0.5: 0.2941  prec@thr=0.5: 0.2941  recall@top3: 0.8235  prec@top3: 0.2745  recall@top5: 0.9412  prec@top5: 0.1882  loss_action_cls: 0.6166
01/08 08:56:48 - mmengine - INFO - Saving checkpoint at 14 epochs
==> 0.0001688 seconds to Reading GT results
==> 5.126e-05 seconds to Reading Detection results
==> 0.897568 seconds to Calculating TP/FP
==> 0.000484228 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
01/08 08:56:51 - mmengine - INFO - Epoch(val) [14][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0093  time: 0.0271
01/08 08:56:53 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:53 - mmengine - INFO - Epoch(train) [15][4/4]  lr: 1.0000e-02  eta: 0:00:08  time: 0.4056  data_time: 0.1018  memory: 18417  grad_norm: 1.8077  loss: 0.5901  recall@thr=0.5: 0.1765  prec@thr=0.5: 0.1765  recall@top3: 0.7941  prec@top3: 0.2745  recall@top5: 0.9118  prec@top5: 0.1882  loss_action_cls: 0.5901
01/08 08:56:53 - mmengine - INFO - Saving checkpoint at 15 epochs
==> 0.000218391 seconds to Reading GT results
==> 5.05447e-05 seconds to Reading Detection results
==> 0.897794 seconds to Calculating TP/FP
==> 0.000442505 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
01/08 08:56:56 - mmengine - INFO - Epoch(val) [15][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0084  time: 0.0248
01/08 08:56:58 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:56:58 - mmengine - INFO - Epoch(train) [16][4/4]  lr: 1.0000e-03  eta: 0:00:06  time: 0.4059  data_time: 0.1029  memory: 18417  grad_norm: 1.1427  loss: 0.5267  recall@thr=0.5: 0.2778  prec@thr=0.5: 0.2778  recall@top3: 0.8889  prec@top3: 0.2963  recall@top5: 0.9444  prec@top5: 0.1889  loss_action_cls: 0.5267
01/08 08:56:58 - mmengine - INFO - Saving checkpoint at 16 epochs
==> 0.000166178 seconds to Reading GT results
==> 5.03063e-05 seconds to Reading Detection results
==> 0.883596 seconds to Calculating TP/FP
==> 0.000499725 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
01/08 08:57:01 - mmengine - INFO - Epoch(val) [16][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0096  time: 0.0267
01/08 08:57:03 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:57:03 - mmengine - INFO - Epoch(train) [17][4/4]  lr: 1.0000e-03  eta: 0:00:05  time: 0.4063  data_time: 0.1031  memory: 18417  grad_norm: 0.8430  loss: 0.4507  recall@thr=0.5: 0.3889  prec@thr=0.5: 0.3611  recall@top3: 0.8889  prec@top3: 0.2963  recall@top5: 1.0000  prec@top5: 0.2000  loss_action_cls: 0.4507
01/08 08:57:03 - mmengine - INFO - Saving checkpoint at 17 epochs
==> 0.000180483 seconds to Reading GT results
==> 6.60419e-05 seconds to Reading Detection results
==> 0.891391 seconds to Calculating TP/FP
==> 0.000629425 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
Index: 6, Action: working: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
Class working AP: 0.0000
01/08 08:57:06 - mmengine - INFO - Epoch(val) [17][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0083  time: 0.0255
01/08 08:57:08 - mmengine - INFO - Exp name: slowfast_demo-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20260108_085528
01/08 08:57:08 - mmengine - INFO - Epoch(train) [18][4/4]  lr: 1.0000e-03  eta: 0:00:03  time: 0.4066  data_time: 0.1037  memory: 18417  grad_norm: 0.8045  loss: 0.4448  recall@thr=0.5: 0.2222  prec@thr=0.5: 0.2222  recall@top3: 0.8333  prec@top3: 0.2778  recall@top5: 0.9444  prec@top5: 0.1889  loss_action_cls: 0.4448
01/08 08:57:08 - mmengine - INFO - Saving checkpoint at 18 epochs
==> 0.000170946 seconds to Reading GT results
==> 8.4877e-05 seconds to Reading Detection results
==> 0.885139 seconds to Calculating TP/FP
==> 0.000639677 seconds to Run Evaluator
Per-class results:
Index: 3, Action: Scan: AP: 0.0000;
Index: 6, Action: working: AP: 0.0000;
Index: 7, Action: Waiting: AP: 0.0000;
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:281: RuntimeWarning: Mean of empty slice
  object_manipulation = np.nanmean([x[2] for x in cls_AP if 14 < x[0] < 64])
/home/user/projects/mmaction2-main/mmaction/evaluation/functional/ava_utils.py:282: RuntimeWarning: Mean of empty slice
  person_interaction = np.nanmean([x[2] for x in cls_AP if 64 <= x[0]])
Overall Results:
Overall mAP: 0.0000
Person Movement mAP: 0.0000
Object Manipulation mAP: nan
Person Interaction mAP: nan
Class Scan AP: 0.0000
Class working AP: 0.0000
Class Waiting AP: 0.0000
01/08 08:57:11 - mmengine - INFO - Epoch(val) [18][8/8]    mAP/overall: 0.0000  mAP/person_movement: 0.0000  mAP/object_manipulation: nan  mAP/person_interaction: nan  data_time: 0.0087  time: 0.0256
