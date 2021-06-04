import utils.gpu as gpu
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import time
import logging
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *
from utils.log import Logger
import os

model_path = 'E:/1.pth'
if os.name!='nt':
    model_path='1.pth'





# 这个代码是为了看效果. 输入一个文件夹,然后把这个文件夹结果都跑出来.然后画框.
class Evaluation(object):
    def __init__(
        self,
        gpu_id=0,
        weight_path=None,
        visiual=None,
        eval=False,
        mode=None
    ):
        self.__num_class = cfg.VOC_DATA["NUM"]
        self.__conf_threshold = cfg.VAL["CONF_THRESH"]
        self.__nms_threshold = cfg.VAL["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__showatt = cfg.TRAIN["showatt"]
        self.__visiual = visiual
        self.__mode = mode
        self.__classes = cfg.VOC_DATA["CLASSES"]

        self.__model = Build_Model(showatt=self.__showatt).to(self.__device)

        self.__model=torch.load(model_path)

        self.__evalter = Evaluator(self.__model, showatt=self.__showatt)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

    def val(self):
        # global logger
        # print("***********Start Evaluation****************")
        start = time.time()
        mAP = 0
        with torch.no_grad():
                APs, inference_time = Evaluator(
                    self.__model, showatt=False
                ).APs_voc()
                for i in APs:
                    # print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                # print("mAP:{}".format(mAP))
                # print("inference time: {:.2f} ms".format(inference_time))
        end = time.time()
        # print("  ===val cost time:{:.4f}s".format(end - start))

    def detection(self):
        # global logger
        if self.__visiual:
            imgs = os.listdir(self.__visiual)
            # print("***********Start Detection****************")
            for v in imgs:
                path = os.path.join(self.__visiual, v)
                # print("val images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                bboxes_prd = self.__evalter.get_bbox(img, v, mode=self.__mode)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(
                        image=img,
                        boxes=boxes,
                        labels=class_inds,
                        probs=scores,
                        class_labels=self.__classes,
                    )
                    path = os.path.join(
                        cfg.PROJECT_PATH, "detection_result/{}".format(v)
                    )
                    print('写入',path)
                    print(img)
                    cv2.imwrite(path, img)
                    # print("saved images : {}".format(path))


if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/best.pt",
        help="weight file path",
    )
    parser.add_argument(
        "--log_val_path", type=str, default="log_val", help="val log file path"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)",
    )
    parser.add_argument(
        "--visiual",
        type=str,
        default="val",
        help="det data path or None",
    )
    parser.add_argument("--mode", type=str, default="det", help="val or det")
    opt = parser.parse_args()

    opt.weight_path=model_path




    if opt.mode == "val":
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            mode=opt.mode
        ).val()
    else:
        Evaluation(
            gpu_id=opt.gpu_id,
            weight_path=opt.weight_path,
            visiual=opt.visiual,
            mode=opt.mode
        ).detection()
