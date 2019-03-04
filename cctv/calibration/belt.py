import math
import numpy as np
from scipy.ndimage import map_coordinates
from skimage import transform
import cv2


class BeltAlignment (object):
    """
    Belt Alignment

    :ivar persp_matrix: a projection matrix that maps one view of a belt to another view of the same belt; usually
        mapping a belt from a view in a fishing video to the view in a calibration video

    """
    def __init__(self, persp_matrix):
        """
        Constructor

        :param persp_matrix: perspective matrix as a (3,3) NumPy array
        """
        self.persp_matrix = persp_matrix

    def projective_transform(self):
        """
        Construct a skimage.transform.ProjectiveTransform that can be used to apply this alignment transformation

        :return: `skimage.transform.ProjectiveTransform` instance
        """
        return transform.ProjectiveTransform(self.persp_matrix)

    def to_json(self):
        """
        Convert to JSON format for serialisation
        :return: JSON data
        """
        return dict(persp_matrix=self.persp_matrix.tolist())

    @staticmethod
    def from_json(js):
        """
        Convert JSON format to a `BeltAlignment` instance

        :param js: JSON data
        :return: a `BeltAlignment` instance
        """
        persp_matrix = np.array(js['persp_matrix'])
        return BeltAlignment(persp_matrix)

    @staticmethod
    def from_points(source, target):
        """
        Estimate from a set of points in a source (calibration) image and target (fishing) image.

        :param source: points in a `(N,2)` NumPy array
        :param target: points in a `(N,2)` NumPy array
        :return: a `BeltAlignment` instance
        """
        xform = transform.ProjectiveTransform()
        xform.estimate(source[:, ::-1], target[:, ::-1])
        return BeltAlignment(xform.params)

    @staticmethod
    def load_json_points(js):
        """
        Load points from JSON data

        Format:
        {
            "belt_alignment": [
                {"name": "name of point pair for readability",
                 "source": {"x": 0.0, "y": 1.0},
                 "target": {"x": 2.0, "y": 10.0}
                 },
                ....
            ]
        }

        :param js: JSON data
        :return: `(source_points, target_points)` where `source_points` and `target_points` are `(N,2)` arrays
        """
        align_js = js['belt_alignment']
        sources = []
        targets = []
        for pair in align_js:
            src_js = pair['source']
            tgt_js = pair['target']
            sources.append([float(src_js['y']), float(src_js['x'])])
            targets.append([float(tgt_js['y']), float(tgt_js['x'])])

        sources = np.array(sources)
        targets = np.array(targets)
        return sources, targets

    @staticmethod
    def from_json_points(js):
        """
        Load points from JSON data and estiamte
        :param js: JSON data
        :return: a `BeltAlignment` instance
        """
        return BeltAlignment.from_points(*BeltAlignment.load_json_points(js))



class BeltCalibration (object):
    """
    Belt calibration

    :ivar xform: a `skimage.transform.ProjectiveTransform` that transforms from camera space to belt space
    :ivar output_image_shape: the shape of the output image
    """
    def __init__(self, persp_matrix=None, rotation=0.0, flip_h=False,
                 phys_crop_y0=0.0, phys_crop_x0=0.0, phys_crop_y1=0.0, phys_crop_x1=0.0,
                 input_image_shape=(800, 1280), pixels_per_metre=900.0, belt_alignment=None):
        if persp_matrix is None:
            persp_matrix = np.eye(3)

        self.persp_matrix = persp_matrix
        self.rotation = rotation
        self.flip_h = flip_h
        self.phys_crop_y0 = phys_crop_y0
        self.phys_crop_x0 = phys_crop_x0
        self.phys_crop_y1 = phys_crop_y1
        self.phys_crop_x1 = phys_crop_x1
        self.input_image_shape = input_image_shape
        self.pixels_per_metre = pixels_per_metre
        self.belt_alignment = belt_alignment

        # Compute the four corners of the input image so that we can determine the bound required to display
        # the complete image in rectilinear space
        input_corners = np.array([
            [0.0, 0.0],
            [input_image_shape[1], 0.0],
            [input_image_shape[1], input_image_shape[0]],
            [0.0, input_image_shape[0]],
        ])

        # Construct the perspective transformation
        persp_xform = transform.ProjectiveTransform(matrix=persp_matrix)

        # Compute rotation and flip transform
        rot_xform = transform.SimilarityTransform(rotation=math.radians(rotation))
        flip_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        if flip_h:
            flip_matrix[0, 0] = -1.0
        flip_xform = transform.AffineTransform(matrix=flip_matrix)

        # Combine perspective, rotation and flip transforms
        persp_rot_xform = flip_xform + rot_xform + persp_xform

        # Transform the input space corners into rectilinear space and compute the bounds
        pixel_range = persp_rot_xform.inverse(input_corners)
        pixel_lower = pixel_range.min(axis=0)
        pixel_upper = pixel_range.max(axis=0)
        rectilinear_shape = (pixel_upper - pixel_lower + 0.5).astype(int)

        # Transform the cropping distances from physical distance to pixel distance
        crop_x0 = int(phys_crop_x0 * pixels_per_metre + 0.5)
        crop_x1 = int(phys_crop_x1 * pixels_per_metre + 0.5)
        crop_y0 = int(phys_crop_y0 * pixels_per_metre + 0.5)
        crop_y1 = int(phys_crop_y1 * pixels_per_metre + 0.5)

        # Top-left crop and cropped shape
        crop_0 = np.array([crop_x0, crop_y0])
        belt_shape = rectilinear_shape - np.array([crop_x0 + crop_x1, crop_y0 + crop_y1])

        # Translation to move top-left of belt to 0,0
        origin_xform = transform.SimilarityTransform(translation=pixel_lower + crop_0)
        # Final transform and output shape
        self.xform = origin_xform + flip_xform + rot_xform + persp_xform
        if self.belt_alignment is not None:
            self.xform = self.xform + self.belt_alignment.projective_transform()
        self.output_image_shape = tuple(belt_shape)[::-1]


    def with_belt_alignment(self, belt_alignment,
                            phys_crop_y0=None, phys_crop_x0=None, phys_crop_y1=None, phys_crop_x1=None):
        if phys_crop_y0 is None:
            phys_crop_y0 = self.phys_crop_y0
        if phys_crop_x0 is None:
            phys_crop_x0 = self.phys_crop_x0
        if phys_crop_y1 is None:
            phys_crop_y1 = self.phys_crop_y1
        if phys_crop_x1 is None:
            phys_crop_x1 = self.phys_crop_x1
        return BeltCalibration(persp_matrix=self.persp_matrix, rotation=self.rotation, flip_h=self.flip_h,
                               phys_crop_y0=phys_crop_y0, phys_crop_x0=phys_crop_x0,
                               phys_crop_y1=phys_crop_y1, phys_crop_x1=phys_crop_x1,
                               input_image_shape=self.input_image_shape, pixels_per_metre=self.pixels_per_metre,
                               belt_alignment=belt_alignment)

    def apply_to_image(self, image):
        """
        Transform an image from camera space to belt space

        :param image:
        :return:
        """
        return transform.warp(image, self.xform, output_shape=self.output_image_shape)

    def rectilinear_mapping(self):
        """
        Compute a distortion rectification map that transforms the belt into rectilinear space.

        Example usage:
        >>> m_x, m_y = belt_calibration.rectilinear_mapping()
        >>> cv2.remap(first_frame, m_x, m_y, cv2.INTER_LINEAR)

        :param lens_calibration:
        :param image_shape:
        :return: tuple `(mapping_x, mapping_y)` where `mapping_x` and `mapping_y` are `(H,W)` arrays that give
        the pixel co-ordinates in the source image to provide the corresponding (h,w) pixel in the destination iamge
        """
        # Pixel co-ordinates in rectilinear space
        rect_y, rect_x = np.meshgrid(np.arange(self.output_image_shape[0]),
                                     np.arange(self.output_image_shape[1]))
        rect_xy = np.append(rect_x[:, :, None], rect_y[:, :, None], axis=2)
        # Transform to camera space
        camera = self.xform(rect_xy.reshape((-1, 2)))
        camera_2d = camera.reshape(rect_xy.shape)
        return camera_2d[:, :, 0].T.astype(np.float32), camera_2d[:, :, 1].T.astype(np.float32)

    def lens_distort_rectilinear_mapping(self, lens_calibration, input_image_shape):
        """
        Compute a distortion rectification map that corrects for lens distortion and transforms the belt into
        rectilinear space in one go.

        Example usage:
        >>> m_x, m_y = belt_calibration.lens_distort_rectilinear_mapping(lens_calibration, (1280, 800))
        >>> cv2.remap(first_frame, m_x, m_y, cv2.INTER_LINEAR)

        :param lens_calibration: a `lens.LensCalibration` instance that provides lens distortion parameters
        :param image_shape: the shape of the input image
        :return: tuple `(mapping_x, mapping_y)` where `mapping_x` and `mapping_y` are `(H,W)` arrays that give
        the pixel co-ordinates in the source image to provide the corresponding (h,w) pixel in the destination iamge
        """
        x, y, w, h = lens_calibration.roi
        rect_x, rect_y = self.rectilinear_mapping()
        lens_map_x, lens_map_y = lens_calibration.distort_rectify_map(input_image_shape)

        belt_map_x = rect_x + float(x)
        belt_map_y = rect_y + float(y)

        query_points = np.append(belt_map_y.flatten()[None, :], belt_map_x.flatten()[None, :], axis=0)
        lens_rect_x = map_coordinates(lens_map_x, query_points, order=1)
        lens_rect_x = lens_rect_x.reshape(belt_map_y.shape).astype(np.float32)

        lens_rect_y = map_coordinates(lens_map_y, query_points, order=1)
        lens_rect_y = lens_rect_y.reshape(belt_map_y.shape).astype(np.float32)

        return lens_rect_x, lens_rect_y

    def lens_undistort_points(self, points, lens_calibration):
        """
        Distort the provided array of points into rectilinear space on the belt

        :param points: points to distort as a `(N,2)` array
        :param lens_calibration: a `lens.LensCalibration` instance that provides lens distortion parameters
        :return: warped points as a `(N,2)` array
        """
        x, y, w, h = lens_calibration.roi
        undistorted_points = lens_calibration.undistort_points(points)
        undistorted_points = undistorted_points - np.array([[x, y]])
        return self.xform.inverse(undistorted_points)

    def to_json(self):
        if self.belt_alignment is not None:
            belt_align_js = self.belt_alignment.to_json()
        else:
            belt_align_js = None
        return dict(persp_matrix=self.persp_matrix.tolist(),
                    rotation=self.rotation,
                    flip_h=self.flip_h,
                    phys_crop_y0=self.phys_crop_y0,
                    phys_crop_x0=self.phys_crop_x0,
                    phys_crop_y1=self.phys_crop_y1,
                    phys_crop_x1=self.phys_crop_x1,
                    input_image_shape=list(self.input_image_shape),
                    belt_align=belt_align_js)


    @staticmethod
    def from_json(calib_js, pixels_per_metre=900.0):
        persp_matrix = np.array(calib_js['persp_matrix'])
        rotation = float(calib_js.get('rotation', 0.0))
        flip_h = bool(calib_js.get('flip_h', False))
        phys_crop_y0 = float(calib_js['phys_crop_y0'])
        phys_crop_x0 = float(calib_js['phys_crop_x0'])
        phys_crop_y1 = float(calib_js['phys_crop_y1'])
        phys_crop_x1 = float(calib_js['phys_crop_x1'])
        input_image_shape = tuple(calib_js.get('input_image_shape', [800, 1280]))[:2]
        belt_align_js = calib_js.get('belt_align')
        if belt_align_js is not None:
            belt_alignment = BeltAlignment.from_json(belt_align_js)
        else:
            belt_alignment = None
        return BeltCalibration(persp_matrix, rotation, flip_h, phys_crop_y0, phys_crop_x0, phys_crop_y1, phys_crop_x1,
                               input_image_shape, pixels_per_metre=pixels_per_metre, belt_alignment=belt_alignment)


class BeltPerspectiveCouldNotFindChessboardError (Exception):
    pass


def belt_perspective_matrix_from_chessboard(image_containing_chessboard, lens_calibration=None, chessboard_cells=(9, 6),
                                            chessboard_interior_physical_size=(0.32, 0.20), corner_subpix_window=(5, 5),
                                            pixels_per_metre=900.0, return_intermediate_results=False):
    """
    Automatically compute the perspective matrix necessary to transform the conveyor belt into rectilinear space.
    The image must contain a chessboard that is lying on the belt. The chessboard will be located and

    :param image_containing_chessboard: an image containing a chessboard, as a (H,W,3) array, with channels in BGR
        order, as per OpenCV
    :param lens_calibration: [optional] a `lens.LensCalibration` object that specifies parameters for correcting the
        camera lens distortion
    :param chessboard_cells: a tuple specifying the number of interior corners in the chessboard, default `(9, 6)`
        for a chessboard with 9x6 corners, or 10x7 cells
    :param chessboard_interior_physical_size: a tuple specifying the physical size (in metres) of the interior cells,
        e.g. for a chessboard that is 10x7 cells in total and is 40cm x 28cm (each cell is 4cm x 4cm), there
        are 9x6 interior corners and 8x5 interior cells, resulting in an interior physical size of 32cm x 20cm
        (the default; `(0.32, 0.20)`)
    :param corner_subpix_window: the window size to use for sub-pixel corner position refinement, default = `(5, 5)`
    :param pixels_per_metre: the number of pixels per metre desired in rectilinear space, default = `900.0`
    :param return_intermediate_results: if `True`, will return chessboard corner positions in the input image
        along with the perspective matrix, if `False` only the perspective matrix will be returned

    :return: if `return_intermediate_results` is `False`:
    - perspective matrix as a (3,3) NumPy array
    if `return_intermediate_results` is `True`:
    the tuple `(perspective_matrix, corners, refined_corners)` where `perspective_matrix` is the perspective
    matrix and `corners` and `refined_corners` are the pixel locations of the interior chessboard corners that
    were found in the form of `(N,2)` arrays.
    """
    corner_subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    img_shape = image_containing_chessboard.shape[:2]

    if lens_calibration is not None:
        # Correct for lens distoirtion
        x, y, w, h = lens_calibration.roi
        mapping_x, mapping_y = lens_calibration.distort_rectify_map(img_shape)
        straight_img = cv2.remap(image_containing_chessboard, mapping_x, mapping_y, cv2.INTER_LINEAR)
        straight_img = straight_img[y:y + h, x:x + w]
    else:
        straight_img = image_containing_chessboard

    straight_shape = straight_img.shape[:2]

    half_shape = straight_shape[0] // 2, straight_shape[1] // 2

    # Convert frame to grey-scale
    grey = cv2.cvtColor(straight_img, cv2.COLOR_BGR2GRAY)

    # Scale to 1/2 size
    half_grey = cv2.resize(grey, half_shape[::-1], interpolation=cv2.INTER_AREA)

    # Find chess board
    chessboard_found, half_corners = cv2.findChessboardCorners(half_grey, chessboard_cells, None)

    if not chessboard_found:
        raise BeltPerspectiveCouldNotFindChessboardError

    # Refine chessboard corners to sub-pixel accuracy
    corners_refined = cv2.cornerSubPix(half_grey, half_corners.copy(), corner_subpix_window, (-1, -1),
                                       corner_subpix_criteria) * 2

    # Compute the locations of the chessboard corners in
    physical_corners = np.ones(chessboard_cells + (2,))
    physical_corners[:, :, 0] *= np.linspace(0.0, chessboard_interior_physical_size[0], chessboard_cells[0])[:, None]
    physical_corners[:, :, 1] *= np.linspace(0.0, chessboard_interior_physical_size[1], chessboard_cells[1])[None, :]
    pixel_corners_list = physical_corners.transpose(1, 0, 2).reshape((-1, 2)) * pixels_per_metre

    # Estimate the perspective transformation
    persp_xform = transform.ProjectiveTransform()
    persp_xform.estimate(pixel_corners_list, corners_refined[:, 0, :])

    if return_intermediate_results:
        return persp_xform.params, half_corners[:,0,:] * 2, corners_refined[:,0,:]
    else:
        return persp_xform.params

