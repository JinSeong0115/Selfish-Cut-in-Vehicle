import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
import cv2
import csv
from shapely.geometry import Point, Polygon
import pandas as pd

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

# 차선 변경 기록을 위한 딕셔너리
lane_history = {}

def initialize_deepsort():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("/home/kim_js/car_project/yolov9/deep_sort_pytorch/configs/deep_sort.yaml")

    model_path = "/home/kim_js/car_project/yolov9/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Please download ckpt.t7.")
    
    deepsort = DeepSort(model_path,
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
    if classid == 0:  # white car
        color = (85, 45, 255)
    elif classid == 2:  # black car
        color = (222, 82, 175)
    elif classid == 3:  # red car
        color = (0, 204, 255)
    elif classid == 5:  # blue car
        color = (0, 149, 255)
    else:
        color = (200, 100, 0)
    return tuple(color)

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

# 현재 차량이 어느 차선에 있는지 판단
def determine_lane(center_x, center_y, lane_polygons):
    point = Point(center_x, center_y)
    for i, lane_polygon in enumerate(lane_polygons):
        if lane_polygon.contains(point):
            return i + 1  # 1차선은 1, 2차선은 2, 3차선은 3
    return None

def group_and_save_csv(input_csv_path, output_csv_path):
    """
    CSV 파일을 로드한 후 id별로 그룹화하고, frame별로 정렬하여 새로운 CSV 파일로 저장합니다.
    """
    # 입력 CSV 파일 읽기
    df = pd.read_csv(input_csv_path)
    
    # id별로 그룹화하고 frame별로 정렬
    grouped_df = df.groupby('id').apply(lambda x: x.sort_values('frame'))
    
    # 중복 제거
    grouped_df = grouped_df.drop_duplicates(subset=['frame', 'id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
    
    # 출력 CSV 파일로 저장
    grouped_df.to_csv(output_csv_path, index=False)
    print(f"Grouped and sorted data saved to {output_csv_path}")

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0,0)):
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
        center = int((x1 + x2) / 2), int((y1 + y2) / 2)

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


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt', 
        source=ROOT / 'data/images', 
        data=ROOT / 'data/coco.yaml', 
        imgsz=(1920, 1088), 
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
        draw_trails=True,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt') 
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source) 

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
    csv_file_path = str(save_dir / 'output.csv')
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    # CSV 컬럼에 center_x, center_y 추가
    csv_writer.writerow(['frame', 'id', 'class', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'center_x', 'center_y', 'conf'])

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
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
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 이미지 크기 관련 스케일링

            # 여기서 리사이즈 수행
            tracking_frame = cv2.resize(im0.copy(), (1920, 1088))  

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], tracking_frame.shape).round()

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
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    bbox_width = abs(x1 - x2)
                    bbox_height = abs(y1 - y2)
                    xcycwh = [cx, cy, bbox_width, bbox_height]
                    xywh_bboxs.append(xcycwh)
                    conf = math.ceil(conf * 100) / 100
                    confs.append(conf)
                    classNameInt = int(cls)
                    oids.append(classNameInt)

                if len(xywh_bboxs) > 0:
                    xywhs = torch.tensor(xywh_bboxs)
                    confss = torch.tensor(confs)

                    if xywhs.ndim == 1:
                        xywhs = xywhs.unsqueeze(0)

                    outputs = deepsort.update(xywhs, confss, oids, tracking_frame)

                    # outputs, identities, object_id의 길이 체크
                    if outputs is not None and len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        object_id = outputs[:, -1]

                        # CSV에 차량 정보 기록 (길이 체크 추가)
                        for j, output in enumerate(outputs):
                            if j < len(identities) and j < len(object_id) and j < len(confs):
                                id = int(identities[j])
                                bbox = bbox_xyxy[j]
                                bbox_x, bbox_y, bbox_w, bbox_h = xyxy2xywh(bbox).tolist()
                                center_x, center_y = int((bbox_x + bbox_w / 2)), int((bbox_y + bbox_h / 2))
                                csv_writer.writerow([frame, id, names[int(object_id[j])], bbox_x, bbox_y, bbox_w, bbox_h, center_x, center_y, confs[j]])
                            else:
                                LOGGER.warning(f"Index out of range: outputs {len(outputs)}, identities {len(identities)}, object_id {len(object_id)}, confs {len(confs)}")

                        # 트래킹된 프레임에 박스 그리기
                        tracking_frame = draw_boxes(tracking_frame, bbox_xyxy, draw_trails, identities, object_id)

            # 저장할 비디오 경로 및 초기화
            if save_img:
                if vid_path[i] != save_path:
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # 기존 비디오 객체 해제
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = tracking_frame.shape[1]
                        h = tracking_frame.shape[0]
                    else:
                        fps, w, h = 30, tracking_frame.shape[1], tracking_frame.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(tracking_frame)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    csv_file.close()

    original_csv_path = csv_file_path
    grouped_csv_path = str(save_dir / 'grouped_output.csv')

    group_and_save_csv(original_csv_path, grouped_csv_path)

    if update:
        strip_optimizer(weights[0])



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0    (webcam)')
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


