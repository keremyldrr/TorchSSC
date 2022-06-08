from seg_opr.metric import hist_info, compute_score
import numpy as np
from sc_utils import export_grid

def print_ssc_iou(sc, ssc):
    lines = []
    type2class = {'empty':0,'cabinet':1, 'bed':2, 'chair':3, 'sofa':4, 'table':5, 'door':6,
            'window':7,'bookshelf':8,'picture':9, 'counter':10, 'desk':11, 'curtain':12,
            'refrigerator':13, 'showercurtrain':14, 'toilet':15, 'sink':16, 'bathtub':17, 'garbagebin':18}  

    revdict={}
    for i,j in type2class.items():
        revdict[j]=i
    names = ["{} {} ".format(revdict[c],sc)  for c,sc in enumerate(ssc[0].tolist()) ]
    lines.append('--*-- Semantic Scene Completion --*--')
    lines.append('IOU: \n{}\n'.format(str(names)))
    lines.append('meanIOU: %f\n' % ssc[2])
    lines.append('pixel-accuracy: %f\n' % ssc[3])
    lines.append('')
    lines.append('--*-- Scene Completion --*--\n')
    lines.append('IOU: %f\n' % sc[0])
    lines.append('pixel-accuracy: %f\n' % sc[1])  # 0 和 1 类的IOU
    lines.append('recall: %f\n' % sc[2])

    line = "\n".join(lines)
    print(line)
    sscmIOU = ssc[2]
    sscPixel = ssc[3]
    scIOU = sc[0]
    scPixel = sc[1]
    scRecall = sc[2]
    return line,sscmIOU, sscPixel,scIOU,scPixel,scRecall
        # return line


def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)  # exclude 255
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))

    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                        minlength=n_cl ** 2).reshape(n_cl,
                                                    n_cl), correct, labeled


def score_to_pred(score):
# print(score.shape,s.
    # sketch_score = sketch_score.permute(1, 2, 3, 0)
    # score = score.permute(1, 2, 3, 0)       # h, w, z, c

    # data_output = score.cpu().numpy()
    # # sketch_output = sketch_score.cpu().numpy()
        
    # pred = data_output.argmax(0)        # 60x36x60
    # pred_sketch = sketch_output.argmax(3)
    """[summary]

    Returns:
        [type]: [description]
    """
    score = score.permute(1, 2, 3, 0)       # h, w, z, c
    

    data_output = score.cpu().numpy()
    
        
    pred = data_output.argmax(3)        # 60x36x60
    
    return pred
def compute_metric(config, results,confusion=False):
    hist_ssc = np.zeros((config.num_classes, config.num_classes))
    correct_ssc = 0
    labeled_ssc = 0

    # scene completion
    tp_sc, fp_sc, fn_sc, union_sc, intersection_sc = 0, 0, 0, 0, 0
    # print("RESULTS",len(results))
    print("Confusion",confusion)
    if confusion:
        conf_mat = np.zeros([config.num_classes,config.num_classes],dtype=np.int64)
    for d in results:
        pred = d['pred'].astype(np.int64)
        label = d['label'].cpu().numpy()[0].astype(np.int64)
        label_weight = d['label_weight'].cpu().numpy()[0].astype(np.float32)
        mapping = d['mapping'].cpu().numpy().astype(np.int64).reshape(-1)
        # print(pred.shape,label.shape,label_weight.shape,mapping.shape)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)

        nonefree =(label_weight > 0)  # Calculate the SSC metric. Exculde the seen atmosphere and the invalid 255 area
        nonefree_pred = flat_pred[nonefree]
        nonefree_label = flat_label[nonefree]
        if confusion:
            for out,targ in zip(nonefree_pred,nonefree_label):
                if targ != 255:
                    # print(out,targ)
                    conf_mat[out,targ] += 1
            
        h_ssc, c_ssc, l_ssc = hist_info(config.num_classes, nonefree_pred, nonefree_label)
        hist_ssc += h_ssc
        correct_ssc += c_ssc
        labeled_ssc += l_ssc
        if config.dataset ==  "ScanNet":
            if config.only_boxes:
                occluded = (label_weight > 0)  # Calculate the SC metric on the occluded area
            else:
        
                occluded =  (label_weight > 0) & (flat_label != 255)   # Calculate the SC metric on the occluded area
        else:
            occluded = (mapping == 307200) & (label_weight > 0) & (flat_label != 255)   # Calculate the SC metric on the occluded area

        res = occluded
        # res2 = nonefree

        grid_shape = [60,36,60]

     
        res2 = np.zeros_like(mapping,dtype=np.int)
        res2[mapping != 307200] = 1
        occluded_pred = flat_pred[occluded]
        occluded_label = flat_label[occluded]
        # export_grid("mppingbox.ply",(res2).reshape(grid_shape).astype(int))        
        # # export_grid("occlbl_sc.ply",(res*flat_label).reshape(grid_shape).astype(int))
        # export_grid("nonefreepred_sc.ply",(res*flat_pred).reshape(grid_shape).astype(int))
        # # export_grid("lblwght_sc.ply",(res2).reshape(grid_shape).astype(int))
   
        # raise  NotADirectoryError
        tp_occ = ((occluded_label > 0) & (occluded_pred > 0)).astype(np.int8).sum()
        fp_occ = ((occluded_label == 0) & (occluded_pred > 0)).astype(np.int8).sum()
        fn_occ = ((occluded_label > 0) & (occluded_pred == 0)).astype(np.int8).sum()

        union = ((occluded_label > 0) | (occluded_pred > 0)).astype(np.int8).sum()
        intersection = ((occluded_label > 0) & (occluded_pred > 0)).astype(np.int8).sum()

        tp_sc += tp_occ
        fp_sc += fp_occ
        fn_sc += fn_occ
        union_sc += union
        intersection_sc += intersection

    score_ssc = compute_score(hist_ssc, correct_ssc, labeled_ssc)
    IOU_sc = intersection_sc / union_sc
    precision_sc = tp_sc / (tp_sc + fp_sc)
    recall_sc = tp_sc / (tp_sc + fn_sc)
    score_sc = [IOU_sc, precision_sc, recall_sc]
    result_line, sscmIOU, sscPixel,scIOU,scPixel,scRecall = print_ssc_iou(score_sc, score_ssc)
    if confusion:
        print(conf_mat)
        conf_mat.tofile("conf_mat.csv",sep=",")
    return result_line,[sscmIOU, sscPixel,scIOU,scPixel,scRecall]
    # result_line = self.print_ssc_iou(score_sc, score_ssc)
    # return result_line
