import json
import cv2
import numpy as np
def search_hp(hp_list,id):
    result = []
    for k in range(len(hp_list)):
        if hp_list[k]['image_id'] == id and hp_list[k]['category_id'] == 1:
            # if hp_list[k]['keypoints'][17] != 0 or hp_list[k]['keypoints'][20] != 0:
                # hp_searched = []
                # hp_searched['hp_id'] = hp_list[k]['id']
            hp_searched = hp_list[k]['keypoints']
            result.append(hp_searched)
    return result

# def search_hp(hp_list,filename):
#     result = []
#     for k in range(len(hp_list)):
#         if hp_list[k]['image_id'] == id and hp_list[k]['category_id'] == 1:
#             if hp_list[k]['keypoints'][17] != 0 and hp_list[k]['keypoints'][20] != 0:
#                 hp_searched = {}
#                 hp_searched['hp_id'] = hp_list[k]['id']
#                 hp_searched['kpt'] = hp_list[k]['keypoints']
#                 result.append(hp_searched)
#     return result

def judge_kpt(kpt_pos):
    if kpt_pos[2] == float(0):
        return '0'
    elif kpt_pos[2] != float(0):
        return kpt_pos[0:2]

def neck_pos(kptpos_list):
    if kptpos_list[17] != 0 and kptpos_list[20] != 0:
        x = (kptpos_list[15] + kptpos_list[18]) / 2
        y = (kptpos_list[16] + kptpos_list[19]) / 2
        return [x,y]
    else:
        return '0'

def convert_pos(kpt_coco):
    kpt_show = []
    kpt_show.append(judge_kpt(kpt_coco[0:3]))#nose
    kpt_show.append(neck_pos(kpt_coco))#neck
    kpt_show.append(judge_kpt(kpt_coco[18:21]))#r_shoulder
    kpt_show.append(judge_kpt(kpt_coco[24:27]))#r_elbow
    kpt_show.append(judge_kpt(kpt_coco[30:33]))#r_wrist
    kpt_show.append(judge_kpt(kpt_coco[15:18]))#l_shoulder
    kpt_show.append(judge_kpt(kpt_coco[21:24]))#l_elbow
    kpt_show.append(judge_kpt(kpt_coco[27:30]))#l_wrist
    kpt_show.append(judge_kpt(kpt_coco[36:39]))#r_hip
    kpt_show.append(judge_kpt(kpt_coco[42:45]))#r_knee
    kpt_show.append(judge_kpt(kpt_coco[48:51]))#r_ankle
    kpt_show.append(judge_kpt(kpt_coco[33:36]))#l_hip
    kpt_show.append(judge_kpt(kpt_coco[39:42]))#l_knee
    kpt_show.append(judge_kpt(kpt_coco[45:48]))#l_ankle
    kpt_show.append(judge_kpt(kpt_coco[6:9]))#r_eye
    kpt_show.append(judge_kpt(kpt_coco[3:6]))#l_eye
    kpt_show.append(judge_kpt(kpt_coco[12:15]))#r_ear
    kpt_show.append(judge_kpt(kpt_coco[9:12]))#l_ear
    return kpt_show

def convert_list(hp_list):
    num_list = len(hp_list)
    result_list = []
    for t in range(num_list):
        hp_buf = convert_pos(hp_list[t])
        result_list.append(hp_buf)
    return result_list
    
def convert_img(file_name,img_list,human_list):
    file_name = file_name[-16:]
    # filename = 'person_keypoints_val2017.json'
    # with open(filename, 'r', encoding='UTF-8') as f:
    #     gt = json.load(f)
    # img_list = gt['images']
    # human_list = gt['annotations']
    num_img = len(img_list)
    global searched_id
    for k in range(num_img):
        if img_list[k]['file_name'] == file_name:
            searched_id = k
            break
    img_dict = {}
    img_dict['file_name'] = img_list[searched_id]['file_name']
    img_dict['height'] = img_list[searched_id]['height']
    img_dict['width'] = img_list[searched_id]['width']
    img_id = img_list[searched_id]['id']
    hp_list = search_hp(human_list,img_id)
    if len(hp_list) == 0:
        return '0'
    else:
        converted_list = convert(hp_list)
        img_dict['human_kpt_list'] = converted_list
        return img_dict
    
def get_kpt_list(img_dict):
    num = len(img_dict['human_kpt_list'])
    kpt_list = []
    for i in range(num):
        kpt_list.append(img_dict['human_kpt_list'][i]['kpt'])
    return kpt_list
    

def get_bbox_single(kpts):
    kpts_valid = []
    for i in range(len(kpts)):
        if kpts[i] != '0':
            kpts_valid.append(kpts[i])
    kpts_valid = np.array(kpts_valid)
    found_keypoints = np.zeros((np.count_nonzero(kpts_valid[:, 0] != -1), 2), dtype=np.int32)
    found_kpt_id = 0
    for kpt_id in range(len(kpts_valid)):
        if kpts_valid[kpt_id, 0] == -1:
            continue
        found_keypoints[found_kpt_id] = kpts_valid[kpt_id]
        found_kpt_id += 1
    bbox = cv2.boundingRect(kpts_valid.astype(int))
    return bbox
