import numpy as np
from skimage.color import hsv2rgb
from skimage.filters import roberts
import cv2

def drawGrid(vis, sqSz):
    """
    drawGrid(vis, sqSz) - draws square grid on image
    input:
        vis - image
        sqSz - size of grid
    output:
        vis - image with grid inserted
    """
    h, w = vis.shape[:2]
    # draw horizontals
    for i in range (sqSz, h-1, sqSz):
        cv2.line(vis, (0, i), (w-1, i), (0, 255, 0))
    # draw verticals
    for i in range (sqSz, w-1, sqSz):
        cv2.line(vis, (i, 0), (i, h-1), (0, 255, 0)) 
    
    return vis

def blend(background, foreground, alpha):
    return background * (1-alpha) + foreground*alpha

def add(background, foreground, alpha):
    return background + foreground*alpha

def visualise_labels_as_rgb(labels):
    hue = np.linspace(0.0, 1.0, labels.max()+1)[:-1]
    np.random.shuffle(hue)
    hue = hue[None,:,None]
    sat_val = np.ones_like(hue)
    hsv = np.concatenate([hue, sat_val, sat_val], axis=2)
    rgb = hsv2rgb(hsv)[0, :, :]

    rgb = np.append(np.array([[0, 0, 0]], dtype=float), rgb, axis=0)
    rgb_labels = rgb[labels]

    # Detect edges
    label_edges = roberts(labels) > 0
    # Edge pixels -> black
    rgb_labels[label_edges, :] = [0, 0, 0]
    return rgb_labels


def label_boundary_mask(labels):
    # Detect edges
    label_edges = roberts(labels) > 0
    return label_edges


class LabelColouriser (object):
    def __init__(self):
        self.label_colour_map = np.array([[0.0, 0.0, 0.0]])


    def colourise(self, labels):
        max_val = labels.max()
        # Allocate new colours if necessary
        if max_val >= len(self.label_colour_map):
            new_len = max_val + 1
            num_entries = new_len - len(self.label_colour_map)

            hues = np.random.uniform(low=0.0, high=1.0, size=(num_entries, 1, 1))
            hsv = np.concatenate([hues, np.ones_like(hues), np.ones_like(hues)], axis=2)
            rgb = hsv2rgb(hsv)
            self.label_colour_map = np.append(self.label_colour_map, rgb[:, 0, :], axis=0)

        # Map to colours
        label_vis = self.label_colour_map[labels]
        # Detect edges
        label_edges = roberts(labels) > 0
        # Edge pixels -> black
        label_vis[label_edges] = [0, 0, 0]
        return label_vis, label_edges
