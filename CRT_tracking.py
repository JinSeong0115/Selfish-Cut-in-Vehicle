import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
import cv2
import csv  # CSV 모듈 추가

# deep_sort_pytorch 모듈 경로 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
DEEPSORT_PATH = ROOT / 'deep_sort_pytorch'
if str(DEEPSORT_PATH) not in sys.path:
    sys.path.append(str(DEEPSORT_PATH))

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

def initialize_deepsort():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, 
                        n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    return deepsort

deepsort = initialize_deepsort()
data_deque = {}

def classNames():
    cocoClassNames = ["white car", "black car", "red car", "blue car"]
    return cocoClassNames

className = classNames()

def colorLabels(classid):
    if classid == 0: # white car
        color = (85, 45, 255)
    elif classid == 2: # black car
        color = (222, 82, 175)
    elif classid == 3: # red car
        color = (0, 204, 255)
    elif classid == 5: # blue car
        color = (0, 149, 255)
    else:
        color = (200, 100, 0)
    return tuple(color)

def is_point_in_polygon(point, polygon):
    """
    point: tuple (x, y)
    polygon: list of tuples [(x1, y1), (x2, y2), ...]
    """
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0,0), roi_points=None):
    height, width, _ = frame.shape
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        y1 += offset[0]
        x2 += offset[0]
        y2 += offset[0]
        center = int((x1+x2)/2), int((y1+y2)/2)

        # ROI 영역 내에 있는지 확인
        if not is_point_in_polygon(center, roi_points):
            continue  # ROI 영역 밖에 있으면 트래킹하지 않음

        cat = int(categories[i]) if categories is not None else 0
        color = colorLabels(cat)
        id = int(identities[i]) if identities is not None else 0
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        data_deque[id].appendleft(center)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = className[cat]
        label = str(id) + ":" + name
        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, 2, (0, 255, 0), cv2.FILLED)
        if draw_trails:
            for j in range(1, len(data_deque[id])):
                if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
                cv2.line(frame, data_deque[id][j - 1], data_deque[id][j], color, thickness)
    return frame

def draw_trail_only(frame, bbox_xyxy, identities=None, offset=(0,0), roi_points=None):
    trail_frame = np.zeros_like(frame)  # 검정 배경 프레임 생성
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        y1 += offset[0]
        x2 += offset[0]
        y2 += offset[0]
        center = int((x1+x2)/2), int((y1+y2)/2)
        
        # ROI 영역 내에 있는지 확인
        if roi_points is not None and not is_point_in_polygon(center, roi_points):
            continue  # ROI 영역 밖에 있으면 트래킹하지 않음

        id = int(identities[i]) if identities is not None else 0
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        data_deque[id].appendleft(center)
        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
            cv2.line(trail_frame, data_deque[id][j - 1], data_deque[id][j], (0, 255, 0), thickness)
    return trail_frame

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt', 
        source=ROOT / 'data/images', 
        data=ROOT / 'data/coco.yaml', 
        imgsz=(1280, 1280), 
        conf_thres=0.25, 
        iou_thres=0.45, 
        max_det=1000, 
        device='', 
        view_img=False, 
        nosave=False, 
        classes=None, 
        agnostic_nms=False, 
        augment=False, 
        visualize=False, 
        update=False, 
        project=ROOT / 'runs/detect', 
        name='exp', 
        exist_ok=False, 
        half=False, 
        dnn=False, 
        vid_stride=1, 
        draw_trails=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt') 
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source) 

    # save_dir을 한 번만 설정합니다.
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok) 
    save_dir.mkdir(parents=True, exist_ok=True) 

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride) 

    bs = 1 
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz)) 
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # CSV 파일 저장을 위한 초기화
    csv_file = open(save_dir / 'output.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'id', 'class', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'conf'])

    # ROI 영역을 지정할 좌표
    roi_points = [
    (1833, 3), (1853, 146), (1870, 316), (1896, 540), 
    (1910, 690), (1933, 800), (1936, 896), (1926, 990), 
    (1876, 1130), (1803, 1250), (1750, 1286), (1630, 1290), 
    (1496, 1296), (1293, 1300), (1030, 1293), (810, 1293), 
    (580, 1290), (356, 1293), (170, 1290), (10, 1290), 
    (0, 1160), (150, 1170), (353, 1166), (573, 1170), 
    (819, 1166), (1026, 1170), (1276, 1170), (1493, 1166), 
    (1653, 1160), (1726, 1156), (1756, 1083), (1773, 976), 
    (1780, 856), (1776, 730), (1763, 570), (1743, 440), 
    (1723, 286), (1703, 156), (1689, 13)
    ]

    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float() 
            im /= 255 
            if len(im.shape) == 3:
                im = im[None] 

        with dt[1]:
            # visualize 경로 설정 시 기존의 save_dir을 사용하도록 수정
            if visualize:
                visualize = save_dir / Path(path).stem
                visualize.mkdir(parents=True, exist_ok=True)
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0]

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred): 
            seen += 1
            if webcam: 
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p) 
            save_path = str(save_dir / p.name) 
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}') 
            s += '%gx%g ' % im.shape[2:] 
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            ims = im0.copy()
            tracking_frame = ims.copy()  # 트래킹된 프레임
            trail_frame = np.zeros_like(ims)  # 경로만 표시할 프레임
            
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum() 
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " 
                xywh_bboxs = []
                confs = []
                oids = []
                outputs = []
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    bbox_width = abs(x1-x2)
                    bbox_height = abs(y1-y2)
                    xcycwh = [cx, cy, bbox_width, bbox_height]
                    xywh_bboxs.append(xcycwh)
                    conf = math.ceil(conf*100)/100
                    confs.append(conf)
                    classNameInt = int(cls)
                    oids.append(classNameInt)
                xywhs = torch.tensor(xywh_bboxs)
                confss = torch.tensor(confs)
                outputs = deepsort.update(xywhs, confss, oids, ims)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    tracking_frame = draw_boxes(tracking_frame, bbox_xyxy, draw_trails, identities, object_id, roi_points=roi_points)
                    trail_frame = draw_trail_only(trail_frame, bbox_xyxy, identities, roi_points=roi_points)

                    # CSV 파일에 기록
                    for j, bbox in enumerate(bbox_xyxy):
                        if j < len(confs) and j < len(identities) and j < len(object_id):
                            bbox_x, bbox_y, bbox_w, bbox_h = xyxy2xywh(bbox).tolist()
                            csv_writer.writerow([frame, identities[j], names[int(object_id[j])], bbox_x, bbox_y, bbox_w, bbox_h, confs[j]])


            # 4개 영상으로 결합하기 전에 크기 조정
            h, w = ims.shape[:2]
            
            # 기존의 비율을 유지하면서 이미지 크기를 조정
            # new_width = imgsz[0]
            new_width = 960
            new_height = int(new_width * h / w)

            ims_resized = cv2.resize(ims, (new_width, new_height))
            tracking_frame_resized = cv2.resize(tracking_frame, (new_width, new_height))
            trail_frame_resized = cv2.resize(trail_frame, (new_width, new_height))

            combined_frame = np.zeros((new_height * 2, new_width * 2, 3), dtype=np.uint8)
            combined_frame[:new_height, :new_width] = ims_resized  # 왼쪽 상단: Original 영상
            combined_frame[:new_height, new_width:2*new_width] = tracking_frame_resized  # 오른쪽 상단: 트래킹된 영상
            combined_frame[new_height:2*new_height, new_width:2*new_width] = trail_frame_resized  # 오른쪽 하단: 경로만 표시된 영상

            if save_img:
                if vid_path[i] != save_path: 
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release() 
                    if vid_cap: 
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = combined_frame.shape[1]  # 새로운 결합된 프레임의 너비
                        h = combined_frame.shape[0]  # 새로운 결합된 프레임의 높이
                    else: 
                        fps, w, h = 30, combined_frame.shape[1], combined_frame.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4')) 
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(combined_frame)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    
    csv_file.close()  # CSV 파일 닫기
    if update:
        strip_optimizer(weights[0])



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--draw-trails', action='store_true', help='draw trails')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1 
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
