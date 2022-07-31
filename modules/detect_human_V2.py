from modules.convert_format import *
from modules.evaluate_link import *

def get_rec(bbox):
    # bbox = [x,y,width,height]
    # re  = [left1,top1,right1,bottom1]
    rec = [bbox[0],bbox[1],(bbox[0]+bbox[2]),(bbox[1]-bbox[3])]
    return rec

def calculate_IOU(rec1,rec2):
    """ 计算两个矩形框的交并比
    
    Args:
    	rec1: [left1,top1,right1,bottom1]  # 其中(left1,top1)为矩形框rect1左上角的坐标，(right1, bottom1)为右下角的坐标，下同。
     	rec2: [left2,top2,right2,bottom2]
     	
    Returns: 
    	交并比IoU值
    """
    left_max  = max(rec1[0],rec2[0])
    top_max = max(rec1[1],rec2[1])
    right_min = min(rec1[2],rec2[2])
    bottom_min = min(rec1[3],rec2[3])
    #两矩形相交时计算IoU
    if (left_max < right_min or bottom_min > top_max):  # 判断时加不加=都行，当两者相等时，重叠部分的面积也等于0
        rect1_area = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        rect2_area = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        area_cross = (bottom_min - top_max)*(right_min - left_max)
        return area_cross / (rect1_area + rect2_area - area_cross)
    else:
        return 0

def cmp_body(IOU,threshold):
    #gt,pt: rec
    # IOU = calculate_IOU(gt,pt)
    # print(IOU)
    if IOU > threshold:
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
                cost_body = calculate_IOU(pt[i],gt[j])
                cost_list.append(cost_body)
            max_pos = cost_list.index(max(cost_list))
            cost_selected = cost_list[max_pos]
            gt.pop(max_pos)
            max_num = max_num + cmp_body(cost_selected,threshold)

    elif num_gt > num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_pt):
            cost_list = []
            for j in range(len(gt)):
                cost_body = calculate_IOU(pt[i],gt[j])
                cost_list.append(cost_body)
            max_pos = cost_list.index(max(cost_list))
            cost_selected = cost_list[max_pos]
            gt.pop(max_pos)
            max_num = max_num + cmp_body(cost_selected,threshold)
    
    elif num_gt < num_pt and num_gt != 0 and num_pt != 0:
        for i in range(num_gt):
            cost_list = []
            for j in range(len(pt)):
                cost_body = calculate_IOU(gt[i],pt[j])
                cost_list.append(cost_body)
            max_pos = cost_list.index(max(cost_list))
            cost_selected = cost_list[max_pos]
            pt.pop(max_pos)
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
        gt_rec_dict = []
        pt_rec_dict = []
        for j in range(len(gt_hp_searched)):
            gt_rec_dict.append(get_rec(get_bbox_single(gt_hp_searched[j])))
        for t in range(len(pt_hp_searched)):
            pt_rec_dict.append(get_rec(get_bbox_single(pt_hp_searched[t])))
        num_all = len(gt_rec_dict)
        right_num = cmp_img(gt_rec_dict,pt_rec_dict,threshold)
        acc_img = right_num / num_all
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