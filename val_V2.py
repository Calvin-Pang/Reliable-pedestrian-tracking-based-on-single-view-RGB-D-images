import argparse
import cv2
import json
import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import time
from datasets.coco import CocoValDataset
from models.with_mobilenet_V1 import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.convert_format import convert_pos,convert_img,get_kpt_list,get_bbox_single
BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
set_color = [0, 224, 255]
set_color_2 = [112,0,128]
set_color_3 = [128,0,64]
color_list = [[0,255,128],[0,0,128],[0,128,0],[128,0,0],[0,0,255],[0,255,0],[255,0,0],[0,128,255],[128,0,255],
         [128,255,0],[128,0,128],[128,128,0],[0,128,128],[255,255,0],[255,0,255],[0,255,255],[128,128,128]]

def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        start_time = time.time()
        stages_output = net(tensor_img)
        end_time = time.time()
        time_used = end_time - start_time
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs,time_used


def evaluate(labels, output_name, images_folder, net, img_input = 3,multiscale=False, visualize=False, img_save_folder='/content/drive/MyDrive/'):
    with open(labels, 'r', encoding='UTF-8') as f:
        gt = json.load(f)
    human_list = gt['annotations']
    net = net.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    img_list = dataset._labels['images']
    coco_result = []
    time_list = []
    for idx,sample in enumerate(dataset):
        file_name = sample['file_name']
        # print(file_name)
        img = sample['img']
        img_show = sample['img_show']
        # image_idx = idx
        
        if img_input == 3:
            avg_heatmaps, avg_pafs, time_used= infer(net, img, scales, base_height, stride, img_mean=(128, 128, 128))
        elif img_input == 4:
            avg_heatmaps, avg_pafs, time_used = infer(net, img, scales, base_height, stride, img_mean=(128, 128, 128,128))
        time_list.append(time_used)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        # image_id = int(file_name[0:file_name.rfind('.')])
        image_id = img_list[idx]['id']
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                # color = np.random.randint(0, 255, 3, dtype=np.int32)
                converted_pos = convert_pos(keypoints)
                if converted_pos[1] != '1':
                    # print('1',converted_pos)
                    bbox = get_bbox_single(converted_pos)
                    cv2.rectangle(img_show, (bbox[0], bbox[1]),(bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0))
                    for idx in range(len(converted_pos)):
                        if converted_pos[idx] != '0':
                            cv2.circle(img_show, (int(converted_pos[idx][0]),int(converted_pos[idx][1])),2, (set_color[0], set_color[1], set_color[2]), -1)
                            # cv2.putText(img_show, str(image_idx), (50,300),cv2.FONT_HERSHEY_PLAIN,5,(int(set_color[0]), int(set_color[1]), int(set_color[2])))
                    for n in range(len(BODY_PARTS_KPT_IDS) - 2):
                        link_id = BODY_PARTS_KPT_IDS[n]
                        start_id = link_id[0]
                        end_id = link_id[1]
                        if converted_pos[start_id] != '0' and converted_pos[end_id] != '0':
                            x_a = converted_pos[start_id][0]
                            y_a = converted_pos[start_id][1]
                            x_b = converted_pos[end_id][0]
                            y_b = converted_pos[end_id][1]
                            cv2.line(img_show, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (int(color_list[n][0]), int(color_list[n][1]), int(color_list[n][2])), 1, lineType = cv2.LINE_AA)
            output = img_save_folder + str(image_id) + '.png'
            # print(type(img))
            cv2.imwrite(output,img_show)
            # print(output)
            # plt.figure(figsize=(20,20))
            # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # plt.show()

            # img_dict = convert_img(file_name,img_list,human_list)
            # if img_dict != '0':
            #     kpt_list = get_kpt_list(img_dict)
            #     # print('kpt_list',kpt_list)
            #     for keypoints in kpt_list:
            #         # converted_pos = convert_pos(keypoints)
            #         if keypoints[1] != '0':
            #             for idx in range(len(keypoints)):
            #                 if keypoints[idx] != '0':
            #                     cv2.circle(img, (int(keypoints[idx][0]),int(keypoints[idx][1])),2, (set_color_2[0], set_color_2[1], set_color_2[2]), -1)
            #             for n in range(len(BODY_PARTS_KPT_IDS) - 2):
            #                 link_id = BODY_PARTS_KPT_IDS[n]
            #                 start_id = link_id[0]
            #                 end_id = link_id[1]
            #                 if keypoints[start_id] != '0' and keypoints[end_id] != '0':
            #                     x_a = keypoints[start_id][0]
            #                     y_a = keypoints[start_id][1]
            #                     x_b = keypoints[end_id][0]
            #                     y_b = keypoints[end_id][1]
            #                     cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (int(set_color_3[0]), int(set_color_3[1]), int(set_color_3[2])), 1, lineType = cv2.LINE_AA)
            # # cv2.putText(img, str(image_idx), (20,50),cv2.FONT_HERSHEY_PLAIN,3,(int(set_color[0]), int(set_color[1]), int(set_color[2])),3)
            # output = img_save_folder + str(image_idx) + '.png'
            # cv2.imwrite(output,img)
            # cv2.imshow('keypoints', img)
            # key = cv2.waitKey()
            # if key == 27:  # esc
            #     return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)
    print('time per image in network:',sum(time_list)/len(time_list))
    run_coco_eval(labels, output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--num-img-input', type=int, default=3, help='number of input(RGB or RGBD)')
    parser.add_argument('--num-refinement-stages', type=int, default=1, help='number of refinement stages')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    parser.add_argument('--img-save-folder', type=str, required=True, help='path to the folder of imgs detected')
    
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet(num_refinement_stages = args.num_refinement_stages)
    # print(type(net))
    # net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    evaluate(args.labels, args.output_name, args.images_folder, net,args.num_img_input, args.multiscale, args.visualize,args.img_save_folder)