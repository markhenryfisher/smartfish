import numpy as np
from skimage.color import hsv2rgb
from skimage.filters import roberts
import cv2

def write_ply(fn, verts, colors):
    
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def montage(n,m,size,*args):
    """
    motage - combine multiple images.
    Bugs - untested for > 2 x 2 
    size = size of combined result
    n = number of (sub-image) rows 
    m = number of (sub-image) cols
    """

    w_final,h_final = size
    h = h_final // n
    w = w_final // m
    
    
    # check and resize the individual images
    if len(args) != (n*m):
        raise ValueError('Insufficiant images to montage')
    
    # assemble images as a montage
    count_i = 0
    for i in range(n):
        for j in range(m):
            img = cv2.resize(args[count_i], (w,h))
            count_i += 1
            if j==0:
                imstack_h = img
            else:
                imstack_h = np.hstack((imstack_h, img))
            
        if i==0:
            imstack_v = imstack_h
        else:
            imstack_v = np.vstack((imstack_v, imstack_h))
        
    return imstack_v
        
    
def overlay(foreground, background):
    """
    highlights foreground with background channel green
    """
    dst = np.uint8(rescale(foreground, (0,127)))
    
    background = rescale(background, (128, 255))
    dst[:,:,1] = np.uint8(rescale(dst[:,:,1] + background, (0, 255)))

    return dst

def anaglyph(left, right):
    """
    makes simple red/blue anaglyph (conventionally they are red/cyan)
    """ 
    left_grey = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_grey = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)  
    dst = left.copy()

    dst[:,:,2] = left_grey
    dst[:,:,0] = right_grey
    dst[:,:,1] = np.zeros_like(right_grey)
    
    return dst

def translateImg(img, delta):
    """
    translateImg - shift image by delta (dx, dy)
    """
    
    dx, dy = delta
    rows,cols = img.shape[:2]

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
       
    return dst

def rescale(src, bounds, *args):
    """
    rescale - rescale data in (a,b)
    y = rescale(x,bounds, minMax)
    Note: minMax is an optional argument that sets m and M; this allows scalefactor 
    to be set independently.
    
    """
    x = np.float64(src.copy())
    a, b = bounds
    if len(args)>0:
        m, M = args[0]
        x = np.clip(x, m, M)
    else:
        m = np.min(x)
        M = np.max(x)
    
    if M-m < np.finfo(np.float64).eps:
        y = x
    else:
        y = (b-a) * (x-m)/(M-m) + a
        
    return y


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
