from seg_opr.metric import hist_info, compute_score
import numpy as np

def print_ssc_iou(sc, ssc):
    lines = []
    lines.append('--*-- Semantic Scene Completion --*--')
    lines.append('IOU: \n{}\n'.format(str(ssc[0].tolist())))
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
def compute_metric(config, results):
    hist_ssc = np.zeros((config.num_classes, config.num_classes))
    correct_ssc = 0
    labeled_ssc = 0

    # scene completion
    tp_sc, fp_sc, fn_sc, union_sc, intersection_sc = 0, 0, 0, 0, 0
    # print("RESULTS",len(results))
    for d in results:
        pred = d['pred'].astype(np.int64)
        label = d['label'].cpu().numpy()[0].astype(np.int64)
        label_weight = d['label_weight'].cpu().numpy()[0].astype(np.float32)
        mapping = d['mapping'].cpu().numpy().astype(np.int64).reshape(-1)
        # print(pred.shape,label.shape,label_weight.shape,mapping.shape)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)

        nonefree =np.where(label_weight > 0)  # Calculate the SSC metric. Exculde the seen atmosphere and the invalid 255 area
        nonefree_pred = flat_pred[nonefree]
        nonefree_label = flat_label[nonefree]

        h_ssc, c_ssc, l_ssc = hist_info(config.num_classes, nonefree_pred, nonefree_label)
        hist_ssc += h_ssc
        correct_ssc += c_ssc
        labeled_ssc += l_ssc

        occluded = (mapping == 307200) & (label_weight > 0) & (flat_label != 255)   # Calculate the SC metric on the occluded area
        occluded_pred = flat_pred[occluded]
        occluded_label = flat_label[occluded]

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
    return result_line,[sscmIOU, sscPixel,scIOU,scPixel,scRecall]
    # result_line = self.print_ssc_iou(score_sc, score_ssc)
    # return result_line
