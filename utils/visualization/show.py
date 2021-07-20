import cv2
import numpy as np

colors = {'green': (0, 255, 0), 'red': (0, 0, 255)}


def dispalyTruePred(image, true, pred, true_color='green', pred_color='red'):
    """
    Display the gt and pred bbox and label on the stitched image.
    1.plot gt bbox on original image and gt bbox & label on new zero1 image
    2.plot pred bbox on original image and pred bbox & label on new zero2 image
    Args:
        image(np.ndarrays): images
        true(list(list)):each one contains (id, bbox, label)
        pred(list(list)):each one contains (id, bbox, label)

    Returns:
    """
    h, w, c = image.shape
    img = image.copy()
    t_img = np.ones_like(img)*255
    p_img = np.ones_like(img)*255
    for t, p in zip(true, pred):
        t_bbox, t_label = t[1:]
        p_bbox, p_label = p[1:]
        t_bbox = np.array(t_bbox).reshape((-1, 1, 2)).astype(np.int32)
        p_bbox = np.array(p_bbox).reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [t_bbox], True, colors[true_color], thickness=1)
        cv2.polylines(t_img, [t_bbox], True, colors[pred_color] if t_label != p_label else colors[true_color],
                      thickness=1)
        cv2.polylines(p_img, [p_bbox], True, colors[pred_color] if t_label != p_label else colors[true_color],
                      thickness=1)

        # should use PIL Draw to draw chinese text
        cv2.putText(t_img, t_label, t_bbox[3, 0], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=colors[pred_color] if t_label != p_label else colors[true_color],
                    thickness=1)
        cv2.putText(p_img, p_label, p_bbox[3, 0], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=colors[pred_color] if t_label != p_label else colors[true_color],
                    thickness=1)

    concat = np.concatenate((img, t_img, p_img), 1)
    cv2.imshow("hah", concat)
    cv2.waitKey()
    return concat

if __name__ == '__main__':
    import copy

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    true = [[1, [[10, 10, 40, 10, 40, 60, 10, 60]], "hehe"], [2, [[50, 10, 90, 10, 90, 60, 50, 60]], "haha"]]
    pred = copy.deepcopy(true)
    pred[1][2] = "nice"
    dispalyTruePred(image, true, pred)
