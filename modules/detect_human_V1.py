from modules.convert_format import *
from modules.evaluate_link import *

def get_center(bbox):
    # bbox = get_bbox_single(kpts)
    center = [bbox[0] + 0.5*bbox[2], bbox[1] + 0.5*bbox[3]]
    return center


def body_cost(gt,pt):
    #gt,pt: bbox
    scale_gt = math.sqrt(gt[2]**2 + gt[3]**2)
    center_gt = get_center(gt)
    center_pt = get_center(pt)
    cost_body = (math.sqrt((center_gt[0] - center_pt[0])**2 + (center_gt[1] - center_pt[1])**2)) / scale_gt
    return cost_body
        
def cmp_body(cost_body,threshold):
    if cost_body < threshold:
        return 1
    else:
        return 0
        
def cmp_img(gt,pt,threshold):
    #gt,pt: bbox list
    max_num = 0
    num_gt = len(gt)
    num_pt = len(pt)
    if num_gt == num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_pt):
            cost_list = []
            for j in range(len(gt)):
                cost_body = body_cost(pt[i],gt[j])
                cost_list.append(cost_body)
            min_pos = cost_list.index(min(cost_list))
            cost_selected = cost_list[min_pos]
            gt.pop(min_pos)
            max_num = max_num + cmp_body(cost_selected,threshold)

    elif num_gt > num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_pt):
            cost_list = []
            for j in range(len(gt)):
                cost_body = body_cost(pt[i],gt[j])
                cost_list.append(cost_body)
            min_pos = cost_list.index(min(cost_list))
            cost_selected = cost_list[min_pos]
            gt.pop(min_pos)
            max_num = max_num + cmp_body(cost_selected,threshold)
    
    elif num_gt < num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_gt):
            cost_list = []
            for j in range(len(pt)):
                cost_body = body_cost(gt[i],pt[j])
                cost_list.append(cost_body)
            min_pos = cost_list.index(min(cost_list))
            cost_selected = cost_list[min_pos]
            pt.pop(min_pos)
            max_num = max_num + cmp_body(cost_selected,threshold)

    elif gt == [] or pt == []:
        max_num = 0
    
    return max_num

    
def detect_body(gt_json,pt_json,threshold,save = False, save_path = ''):
    with open(gt_json, "r") as f:
        gt_dict = json.load(f)
    with open(pt_json, "r") as f:
        pt_anno_list = json.load(f)
    gt_img_list = gt_dict['images']
    gt_anno_list = gt_dict['annotations']
    acc_list = []
    save_list = []
    for i in range(len(gt_img_list)):
        image_id = gt_img_list[i]['id']
        gt_hp_searched = convert_list(search_hp(gt_anno_list,image_id))
        pt_hp_searched = convert_list(search_hp(pt_anno_list,image_id))
        gt_bbox_dict = []
        pt_bbox_dict = []
        for j in range(len(gt_hp_searched)):
            gt_bbox_dict.append(get_bbox_single(gt_hp_searched[j]))
        for t in range(len(pt_hp_searched)):
            pt_bbox_dict.append(get_bbox_single(pt_hp_searched[t]))
        num_all = len(gt_bbox_dict)
        right_num = cmp_img(gt_bbox_dict,pt_bbox_dict,threshold)
        if right_num != 0 and num_all != 0:
            acc_img = right_num / num_all
        elif right_num ==0 and num_all ==0:
            acc_img = 1
        else:
            acc_img = 0
        acc_list.append(acc_img)
        save_list.append({'image_id':image_id,'right_num:':right_num,'num_all':num_all})
        # print('image_id:',image_id,'  acc_img:{:.4f}'.format(acc_img),right_num,'/',num_all)
    acc_avg_dst = sum(acc_list) / len(acc_list)
    save_list.append({'acc_dst:':acc_avg_dst})
    if save == True:
        json.dump(save_list,open(save_path,'w'))
    # print('\n')
    print('acc_avg:{:.4f}'.format(acc_avg_dst))
    return acc_avg_dst