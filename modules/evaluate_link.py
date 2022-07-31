from modules.convert_format import *
import math
from itertools import permutations
import json

def create_link(k):
    link = {}
    if k[0] != '0' and k[1] != '0':
        link['neck'] = [[k[0][0],k[0][1]],[k[1][0],k[1][1]]]
    else:
        link['neck'] = '0'
    if k[1] != '0' and k[2] != '0':
        link['Rshoulder'] = [[k[1][0],k[1][1]],[k[2][0],k[2][1]]]
    else:
        link['Rshoulder'] = '0'
    if k[2] != '0' and k[3] != '0':
        link['Rupperarm'] = [[k[2][0],k[2][1]],[k[3][0],k[3][1]]]
    else:
        link['Rupperarm'] = '0'
    if k[3] != '0' and k[4] != '0':
        link['Rlowerarm'] = [[k[3][0],k[3][1]],[k[4][0],k[4][1]]]
    else:
        link['Rlowerarm'] = '0'
    if k[1] != '0' and k[5] != '0':
        link['Lshoulder'] = [[k[1][0],k[1][1]],[k[5][0],k[5][1]]]
    else:
        link['Lshoulder'] = '0'
    if k[5] != '0' and k[6] != '0':
        link['Lupperarm'] = [[k[5][0],k[5][1]],[k[6][0],k[6][1]]]
    else:
        link['Lupperarm'] = '0'
    if k[6] != '0' and k[7] != '0':
        link['Llowerarm'] = [[k[6][0],k[6][1]],[k[7][0],k[7][1]]]
    else:
        link['Llowerarm'] = '0'
    if k[14] != '0' and k[16] != '0':
        link['Rchick'] = [[k[14][0],k[14][1]],[k[16][0],k[16][1]]]
    else:
        link['Rchick'] = '0'
    if k[0] != '0' and k[14] != '0':
        link['Rtemple'] = [[k[0][0],k[0][1]],[k[14][0],k[14][1]]]
    else:
        link['Rtemple'] = '0'
    if k[0] != '0' and k[15] != '0':
        link['Ltemple'] = [[k[0][0],k[0][1]],[k[15][0],k[15][1]]]
    else:
        link['Ltemple'] = '0'
    if k[15] != '0' and k[17] != '0':
        link['Lchick'] = [[k[15][0],k[15][1]],[k[17][0],k[17][1]]]
    else:
        link['Lchick'] = '0'
    if k[1] != '0' and k[8] != '0':
        link['Rbodyside'] = [[k[1][0],k[1][1]],[k[8][0],k[8][1]]]
    else:
        link['Rbodyside'] = '0'
    if k[1] != '0' and k[11] != '0':
        link['Lbodyside'] = [[k[1][0],k[1][1]],[k[11][0],k[11][1]]]
    else:
        link['Lbodyside'] = '0'
    if k[8] != '0' and k[9] != '0':
        link['Rupperleg'] = [[k[8][0],k[8][1]],[k[9][0],k[9][1]]]
    else:
        link['Rupperleg'] = '0'
    if k[9] != '0' and k[10] != '0':
        link['Rlowerleg'] = [[k[9][0],k[9][1]],[k[10][0],k[10][1]]]
    else:
        link['Rlowerleg'] = '0'
    if k[11] != '0' and k[12] != '0':
        link['Lupperleg'] = [[k[11][0],k[11][1]],[k[12][0],k[12][1]]]
    else:
        link['Lupperleg'] = '0'
    if k[12] != '0' and k[13] != '0':
        link['Llowerleg'] = [[k[12][0],k[12][1]],[k[13][0],k[13][1]]]
    else:
        link['Llowerleg'] = '0'
    return link
    
def judge_link(gt,pt,threshold):
    # print(gt,pt)
    gt_len = math.sqrt((gt[0][0] - gt[1][0])**2 + (gt[0][1] - gt[1][1])**2)
    cost_start = (math.sqrt((gt[0][0] - pt[0][0])**2 + (gt[0][1] - pt[0][1])**2)) / gt_len
    cost_end = (math.sqrt((gt[1][0] - pt[1][0])**2 + (gt[1][1] - pt[1][1])**2)) / gt_len
    # print(cost_start,cost_end)
    if cost_start < threshold and cost_end < threshold:
        return True
    else:
        return False


def count_link(gt_list):
    num_all = 0
    for i in range(len(gt_list)):
        for key in gt_list[i].keys():
            if gt_list[i][key] != '0':
                num_all += 1
    return num_all

def com_single(gt,pt,threshold):
    num = 0
    for key in gt.keys():
        if gt[key] != '0' and pt[key] != '0' and judge_link(gt[key],pt[key],threshold) == True:
            num = num + 1
            
    return num
            

# find right links in one class
def cmp_link(gt,pt,threshold):
    max_num = 0
    num_gt = len(gt)
    num_pt = len(pt)
    if num_gt == num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_pt):
            num_list = []
            for j in range(len(gt)):
                num_list.append(com_single(pt[i],gt[j],threshold))
            max_pos = num_list.index(max(num_list))
            num_right = num_list[max_pos]
            gt.pop(max_pos)
            max_num = max_num + num_right

    elif num_gt > num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_pt):
            num_list = []
            for j in range(len(gt)):
                num_list.append(com_single(pt[i],gt[j],threshold))
            max_pos = num_list.index(max(num_list))
            num_right = num_list[max_pos]
            gt.pop(max_pos)
            max_num = max_num + num_right
    
    elif num_gt < num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_gt):
            num_list = []
            for j in range(len(pt)):
                num_list.append(com_single(gt[i],pt[j],threshold))
            max_pos = num_list.index(max(num_list))
            num_right = num_list[max_pos]
            pt.pop(max_pos)
            max_num = max_num + num_right

    elif gt == [] or pt == []:
        max_num = 0
    
    return max_num

def evaluate_single(gt_json,pt_json,img_id,threshold):
    with open(gt_json, "r") as f:
        gt_dict = json.load(f)
    with open(pt_json, "r") as f:
        pt_anno_list = json.load(f)
    gt_img_list = gt_dict['images']
    gt_anno_list = gt_dict['annotations']
    # scale = math.sqrt((gt_img_list[img_id]['width'])**2 + (gt_img_list[img_id]['height'])**2) #scale of the image
    gt_hp_searched = convert_list(search_hp(gt_anno_list,img_id))
    pt_hp_searched = convert_list(search_hp(pt_anno_list,img_id))
    gt_link_dict = []
    pt_link_dict = []
    for j in range(len(gt_hp_searched)):
        gt_link_dict.append(create_link(gt_hp_searched[j]))
    for t in range(len(pt_hp_searched)):
        pt_link_dict.append(create_link(pt_hp_searched[t]))
    num_all = count_link(gt_link_dict)
    right_num = cmp_link(gt_link_dict,pt_link_dict,threshold)
    acc_img = right_num / num_all
    print('image_id:',img_id,'  acc_img:{:.4f}'.format(acc_img),right_num,'/',num_all) 

def evaluate_dst(gt_json,pt_json,threshold):
    with open(gt_json, "r") as f:
        gt_dict = json.load(f)
    with open(pt_json, "r") as f:
        pt_anno_list = json.load(f)
    gt_img_list = gt_dict['images']
    gt_anno_list = gt_dict['annotations']
    acc_list = []
    for i in range(len(gt_img_list)):
        image_id = gt_img_list[i]['id']
        # scale = math.sqrt((gt_img_list[i]['width'])**2 + (gt_img_list[i]['height'])**2) #scale of the image
        gt_hp_searched = convert_list(search_hp(gt_anno_list,image_id))
        pt_hp_searched = convert_list(search_hp(pt_anno_list,image_id))
        gt_link_dict = []
        pt_link_dict = []
        for j in range(len(gt_hp_searched)):
            gt_link_dict.append(create_link(gt_hp_searched[j]))
        for t in range(len(pt_hp_searched)):
            pt_link_dict.append(create_link(pt_hp_searched[t]))
        num_all = count_link(gt_link_dict)
        right_num = cmp_link(gt_link_dict,pt_link_dict,threshold)
        if right_num != 0 and num_all != 0:
            acc_img = right_num / num_all
        elif right_num ==0 and num_all ==0:
            acc_img = 1
        else:
            acc_img = 0
        acc_list.append(acc_img)
        # print('image_id:',image_id,'  acc_img:{:.4f}'.format(acc_img),right_num,'/',num_all)
    acc_avg_dst = sum(acc_list) / len(acc_list)
    # print('\n')
    print('acc_avg:{:.4f}'.format(acc_avg_dst))
    return acc_avg_dst