'''


2021-05-28,17点52


下面修改成自己的数据集.




2021-05-16,8点55
今天开整yolov4. 这个改动最大的版本, 感觉比yolo5实用.


笔记直接记录这里吧:
1.下载voc2017数据  . 名字叫做trainval 和test
    我现在的名字是这个:VOCtrainval_06-Nov-2007.tar
                     VOCtest_06-Nov-2007.tar
    解压到了2个目录一个起名voc_trainval_data 一个是voc_test_data
2.运行utils/voc.py
3.然后运行当前这个main
    记住标记的物理含义:
     [xmin, ymin, xmax, ymax, str(class_id)] #这个地方要记住物理含义!!!!!!!!!!!!!!!!


'''
import logging
import utils.gpu as gpu
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
import config.yolov4_myconfig as cfg
from utils import cosine_lr_scheduler
from utils.log import Logger
# from apex import amp
from eval_coco import *
from eval.cocoapi_evaluator import COCOAPIEvaluator
import os
import math
def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), targets


class Trainer(object):
    def __init__(self, weight_path=None,
                 resume=False,
                 gpu_id=0,
                 accumulate=1,
                 fp_16=False):
        init_seeds(0)
        self.fp_16 = fp_16
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.accumulate = accumulate
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.showatt = cfg.TRAIN["showatt"]
        if self.multi_scale_train:
            print("Using multi scales training")
        else:
            print("train img size is {}".format(cfg.TRAIN["TRAIN_IMG_SIZE"]))
        self.train_dataset = data.Build_Dataset(
            anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]
        )
        self.epochs = (
            cfg.TRAIN["YOLO_EPOCHS"]
            if cfg.MODEL_TYPE["TYPE"] == "YOLOv4"
            else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        )
        self.eval_epoch = (
            30 if cfg.MODEL_TYPE["TYPE"] == "YOLOv4" else 50
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN["BATCH_SIZE"],
            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
            shuffle=True,
            pin_memory=True,
        )

        self.yolov4 = Build_Model(weight_path=weight_path, resume=resume, showatt=self.showatt).to(
            self.device
        )

        self.optimizer = optim.SGD(
            self.yolov4.parameters(),
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"],
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"],
        )

        self.criterion = YoloV4Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],
        )

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader),
        )
        if resume:
            self.__load_resume_weights(weight_path)

    def __load_resume_weights(self, weight_path):

        # last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
        last_weight = weight_path
        chkpt = torch.load(last_weight, map_location=self.device)
        self.yolov4.load_state_dict(chkpt["model"])

        self.start_epoch = chkpt["epoch"] + 1
        if chkpt["optimizer"] is not None:
            self.optimizer.load_state_dict(chkpt["optimizer"])
            self.best_mAP = chkpt["best_mAP"]
        del chkpt

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(
            os.path.split(self.weight_path)[0], "best.pt"
        )
        last_weight = os.path.join(
            os.path.split(self.weight_path)[0], "last.pt"
        )
        chkpt = {
            "epoch": epoch,
            "best_mAP": self.best_mAP,
            "model": self.yolov4.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt["model"], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(
                chkpt,
                os.path.join(
                    os.path.split(self.weight_path)[0],
                    "backup_epoch%g.pt" % epoch,
                ),
            )
        del chkpt

    def train(self):
        global writer
        # print(
        #     "Training start,img size is: {:d},batchsize is: {:d},work number is {:d}".format(
        #         cfg.TRAIN["TRAIN_IMG_SIZE"],
        #         cfg.TRAIN["BATCH_SIZE"],
        #         cfg.TRAIN["NUMBER_WORKERS"],
        #     )
        # )
        # print(self.yolov4)
        # print(
        #     "Train datasets number is : {}".format(len(self.train_dataset))
        # )

        def is_valid_number(x):
            return not (math.isnan(x) or math.isinf(x) or x > 1e4)
        if self.fp_16:
            self.yolov4, self.optimizer = amp.initialize(
                self.yolov4, self.optimizer, opt_level="O1", verbosity=0
            )
        # print("        =======  start  training   ======     ")
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.yolov4.train()

            mloss = torch.zeros(4)
            # print("===Epoch:[{}/{}]===".format(epoch, self.epochs))
            for i, (
                imgs,
                label_sbbox,
                label_mbbox,
                label_lbbox,
                sbboxes,
                mbboxes,
                lbboxes,
            ) in enumerate(self.train_dataloader): # 读取图片数据.
                self.scheduler.step(
                    len(self.train_dataloader)
                    / (cfg.TRAIN["BATCH_SIZE"])
                    * epoch
                    + i
                )

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov4(imgs)  # p 里面3个, 第一个shape : 1,52,52,3,25 :第一个batch_size,
#p:  The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]  3个锚点. 每一个grid 我们分配3个锚点. 最后的25: 偏移量4个,然后加置信度,表示是物体的概率.    p 跟p_d一样,但是p 没啥用. p_d才是真正的跟label进行比较的.
                loss, loss_ciou, loss_conf, loss_cls = self.criterion(
                    p,
                    p_d,
                    label_sbbox,
                    label_mbbox,
                    label_lbbox,
                    sbboxes,
                    mbboxes,
                    lbboxes,
                )
                if is_valid_number(loss.item()):
                    if self.fp_16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                # Accumulate gradient for x batches before optimizing
                if i % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update running mean of tracked metrics
                loss_items = torch.tensor(
                    [loss_ciou, loss_conf, loss_cls, loss]
                )
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 10 == 0:

                    print(
                        "  === Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{:3}],total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{:.4f}".format(
                            epoch,
                            self.epochs,
                            i,
                            len(self.train_dataloader) - 1,
                            self.train_dataset.img_size,
                            mloss[3],
                            mloss[0],
                            mloss[1],
                            mloss[2],
                            self.optimizer.param_groups[0]["lr"],
                        )
                    )

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i + 1) % 10 == 0:
                    self.train_dataset.img_size = (
                        random.choice(range(10, 20)) * 32
                    )

            if (
                cfg.TRAIN["DATA_TYPE"] == "VOC"
                or cfg.TRAIN["DATA_TYPE"] == "Customer"
            ):
                mAP = 0.0
                if epoch >= self.eval_epoch:
                    print(
                        "===== Validate =====".format(epoch, self.epochs)
                    )
                    print("val img size is {}".format(cfg.VAL["TEST_IMG_SIZE"]))
                    with torch.no_grad():
                        APs, inference_time = Evaluator(
                            self.yolov4, showatt=self.showatt
                        ).APs_voc()
                        for i in APs:
                            print("{} --> mAP : {}".format(i, APs[i]))
                            mAP += APs[i]
                        mAP = mAP / self.train_dataset.num_classes
                        print("mAP : {}".format(mAP))
                        print(
                            "inference time: {:.2f} ms".format(inference_time)
                        )
                        writer.add_scalar("mAP", mAP, epoch)
                        self.__save_model_weights(epoch, mAP)
                        print("save weights done")
                    print("  ===test mAP:{:.3f}".format(mAP))
            elif epoch >= 0 and cfg.TRAIN["DATA_TYPE"] == "COCO":
                evaluator = COCOAPIEvaluator(
                    model_type="YOLOv4",
                    data_dir=cfg.DATA_PATH,
                    img_size=cfg.VAL["TEST_IMG_SIZE"],
                    confthre=0.08,
                    nmsthre=cfg.VAL["NMS_THRESH"],
                )
                ap50_95, ap50 = evaluator.evaluate(self.yolov4)
                print("ap50_95:{}|ap50:{}".format(ap50_95, ap50))
                writer.add_scalar("val/COCOAP50", ap50, epoch)
                writer.add_scalar("val/COCOAP50_95", ap50_95, epoch)
                self.__save_model_weights(epoch, ap50)
                print("save weights done")
            end = time.time()
            print("  ===cost time:{:.4f}s".format(end - start))
        print(
            "=====Training Finished.   best_test_mAP:{:.3f}%====".format(
                self.best_mAP
            )
        )


if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/mobilenetv2.pth",
        help="weight file path",
    )  # weight/darknet53_448.weights
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training flag",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="whither use GPU(0) or CPU(-1)",
    )#默认不实用gpu
    parser.add_argument("--log_path", type=str, default="log/", help="log path")
    parser.add_argument(
        "--accumulate",
        type=int,
        default=2,
        help="batches to accumulate before optimizing",
    )
    parser.add_argument(
        "--fp_16",
        type=bool,
        default=False,
        help="whither to use fp16 precision",
    )
    opt = parser.parse_args()
    # writer = SummaryWriter(logdir=opt.log_path + "/event")
    # logger = Logger(
    #     log_file_name=opt.log_path + "/log.txt",
    #     log_level=logging.DEBUG,
    #     logger_name="YOLOv4",
    # ).get_log()

    Trainer(
        weight_path=opt.weight_path,
        resume=opt.resume,
        gpu_id=opt.gpu_id,
        accumulate=opt.accumulate,
        fp_16=opt.fp_16,
    ).train()
