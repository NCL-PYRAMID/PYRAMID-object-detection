from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
import numpy as np
from tqdm import tqdm
import DOTA_devkit.polyiou as polyiou
import math
import pdb
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class DetectorModel():
    def __init__(self,
                 config_file,
                 checkpoint_file):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        self.classnames = self.dataset.CLASSES
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def inference_single(self, imagname, slide_size, chip_size):
        img = mmcv.imread(imagname)
        height, width, channel = img.shape
        slide_h, slide_w = slide_size
        hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()
        total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]

        for i in tqdm(range(int(width / slide_w + 1))):
            for j in range(int(height / slide_h) + 1):
                subimg = np.zeros((hn, wn, channel))
                # print('i: ', i, 'j: ', j)
                chip = img[j*slide_h:j*slide_h + hn, i*slide_w:i*slide_w + wn, :3]
                subimg[:chip.shape[0], :chip.shape[1], :] = chip

                chip_detections = inference_detector(self.model, subimg)

                # print('result: ', result)
                for cls_id, name in enumerate(self.classnames):
                    chip_detections[cls_id][:, :8][:, ::2] = chip_detections[cls_id][:, :8][:, ::2] + i * slide_w
                    chip_detections[cls_id][:, :8][:, 1::2] = chip_detections[cls_id][:, :8][:, 1::2] + j * slide_h
                    # import pdb;pdb.set_trace()
                    try:
                        total_detections[cls_id] = np.concatenate((total_detections[cls_id], chip_detections[cls_id]))
                    except:
                        import pdb; pdb.set_trace()
        # nms
        for i in range(len(self.classnames)):
            keep = py_cpu_nms_poly_fast_np(total_detections[i], 0.1)
            total_detections[i] = total_detections[i][keep]
        return total_detections

    def inference_single_vis(self, srcpath, dstpath, slide_size, chip_size):
        detections = self.inference_single(srcpath, slide_size, chip_size)
        img = draw_poly_detections(srcpath, detections, self.classnames, scale=1, threshold=0.3)
        id = srcpath[-11:-5]
        for i in range(len(self.classnames)):
            with open(os.path.join(dstpath[:-12], "BboxAndScore_{}_{}.txt".format(id, i)), "w") as output:
                for item in detections[i]:
                    output.write("%s\n" % str(item)[1:-1])
                # output.write(str(detections))
        with open(os.path.join(dstpath[:-12], "classnames.txt"), "w") as op:
            op.write(str(self.classnames) [1:-1])
        cv2.imwrite(dstpath, img)


if __name__ == '__main__':

    current_path = os.getcwd()
    orthoimage_path = os.path.join(current_path, 'orthoimages_tif/')

    if os.path.exists(str(os.path.join(current_path, 'orthoimages_jpeg/'))):
        output_orthoimage_path = os.path.join(current_path, 'orthoimages_jpeg/')
    else:
        output_orthoimage_path = os.mkdir(os.path.join(current_path, 'orthoimages_jpeg/'))

    for root, dirs, files in os.walk(orthoimage_path, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            # if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
                if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpeg"):
                    print("A jpeg file already exists for %s" % name)
                # If a jpeg with the name does *NOT* exist, covert one from the tif.
                else:
                    outputfile = os.path.splitext(os.path.join(output_orthoimage_path, name))[0] + ".jpeg"
                    im = Image.open(os.path.join(root, name))
                    print("Converting jpeg for %s" % name)
                    im.thumbnail(im.size)
                    im.save(outputfile, "JPEG", quality=100)

    roitransformer_dota_1_0 = DetectorModel(r'configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py',
                                            r'PYRAMID pre-trained object detection model/dota10.pth')
    roitransformer_dota_1_5 = DetectorModel(r'configs/DOTA1_5/faster_rcnn_RoITrans_r50_fpn_1x_dota1_5.py',
                                            r'PYRAMID pre-trained object detection model/dota15.pth')

    if os.path.exists(str(os.path.join(current_path, 'dota_1_0_res/'))):
        dota_1_0_res = os.path.join(current_path, 'dota_1_0_res/')
    else:
        dota_1_0_res = os.mkdir(os.path.join(current_path, 'dota_1_0_res/'))

    if os.path.exists(str(os.path.join(current_path, 'dota_1_5_res/'))):
        dota_1_5_res = os.path.join(current_path, 'dota_1_5_res/')
    else:
        dota_1_5_res = os.mkdir(os.path.join(current_path, 'dota_1_5_res/'))

    for imgnames in os.walk(os.path.join(current_path, 'orthoimages_jpeg/'), topdown=False):
        for i, img in enumerate(list(imgnames[2])):
            img_id = os.path.join(current_path, 'orthoimages_jpeg/', img)
            dota_1_0_out_img_id = os.path.join(dota_1_0_res, img)
            dota_1_5_out_img_id = os.path.join(dota_1_5_res, img)
            roitransformer_dota_1_0.inference_single_vis(img_id, dota_1_0_out_img_id, (512, 512), (1024, 1024)) ### (800, 800), (1024, 1024)
            roitransformer_dota_1_5.inference_single_vis(img_id, dota_1_5_out_img_id, (512, 512), (1024, 1024)) ### (800, 800), (1024, 1024)
    print('Detection Finished!')


