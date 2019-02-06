import numpy as np
import cv2
from utils import video_seek


class CalibrationError (Exception):
    pass

class CalibrationCouldNotFindChessboardError (CalibrationError):
    pass

class CalibrationFailedError (CalibrationError):
    pass



class LensCalibration (object):
    def __init__(self, camera_matrix, dist_coeffs, new_camera_matrix, roi):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.new_camera_matrix = new_camera_matrix
        self.roi = roi

    def distort_rectify_map(self, image_shape):
        mapping_x, mapping_y = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix,
            image_shape[::-1], cv2.CV_32FC1)
        return mapping_x, mapping_y

    def undistort_points(self, points):
        return cv2.undistortPoints(points[:, None, :].astype(np.float32), self.camera_matrix, self.dist_coeffs,
                                   None, None, self.new_camera_matrix)

    def to_json(self):
        return dict(camera_matrix=self.camera_matrix.tolist(),
                    dist_coeffs=self.dist_coeffs.tolist(),
                    new_camera_matrix=self.new_camera_matrix.tolist(),
                    roi=list(self.roi))

    @staticmethod
    def from_json(calib_js):
        camera_matrix = np.array(calib_js['camera_matrix'])
        dist_coeffs = np.array(calib_js['dist_coeffs'])
        new_camera_matrix = np.array(calib_js['new_camera_matrix'])
        roi = tuple(calib_js['roi'])
        return LensCalibration(camera_matrix, dist_coeffs, new_camera_matrix, roi)


def calibrate_lens_auto(cap, grid_size=(9, 6), hist_cell_size=(50, 50), coverage_threshold=0.2222,
                        progress_bar_fn=None):
    corner_subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    obj_p[:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1,2)

    obj_points = [] # 3d point in real world space
    img_points = [] # 2d points in image plane.

    # Load the video
    video_seek.video_seek_start(cap)

    # Get the image shape
    img_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    half_shape = img_shape[0] // 2, img_shape[1] // 2
    qtr_shape = img_shape[0] // 4, img_shape[1] // 4

    # Setup for coverage
    cov_hist_shape = img_shape[0] // hist_cell_size[0], img_shape[1] // hist_cell_size[1]
    cov_hist_bins_x = np.linspace(0, qtr_shape[1], cov_hist_shape[1] + 1)
    cov_hist_bins_y = np.linspace(0, qtr_shape[0], cov_hist_shape[0] + 1)
    cumulative_coverage = np.zeros(cov_hist_shape, dtype=bool)
    coverage_thresh_n = int(float(grid_size[0] * grid_size[1]) * coverage_threshold)

    if progress_bar_fn is not None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = progress_bar_fn(total=num_frames)
    else:
        pbar = None
    while True:
        # Read a frame
        success, img = cap.read()

        if success:
            # Convert frame to grey-scale
            grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Scale to 1/4 size
            qtr_grey = cv2.resize(grey, qtr_shape[::-1], interpolation=cv2.INTER_AREA)

            # Find chess board
            ret, qtr_corners = cv2.findChessboardCorners(qtr_grey, grid_size, None)

            if ret:
                # Chessboard found
                # Compute 2D histogram of the co-ordinates of the corners that were found
                hist_2d, _, _ = np.histogram2d(qtr_corners[:,0,1], qtr_corners[:,0,0],
                                               bins=(cov_hist_bins_y, cov_hist_bins_x))

                chessboard_coverage = hist_2d > 0

                if (chessboard_coverage & ~cumulative_coverage).sum() > coverage_thresh_n:
                    # Find the chess board corners on the half-res image
                    half_grey = cv2.resize(grey, half_shape[::-1], interpolation=cv2.INTER_AREA)
                    ret, corners = cv2.findChessboardCorners(half_grey, grid_size, None)

                    # If found, add object points, image points (after refining them)
                    if ret == True:
                        # Add object co-ordinates
                        obj_points.append(obj_p)

                        # Refine chessboard corners to sub-pixel accuracy
                        corners2 = cv2.cornerSubPix(half_grey, corners, (11,11), (-1,-1), corner_subpix_criteria)

                        # Scale up to full res and add to image points
                        img_points.append(corners2*2)

                        # Update coverage map
                        cumulative_coverage = chessboard_coverage | cumulative_coverage
        else:
            break

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Remove a dimension
    img_points = [c[:,0,:] for c in img_points]

    if len(img_points) == 0:
        raise CalibrationCouldNotFindChessboardError

    # Compute lens calibration
    ret, camera_matrix, dist_coeffs, rot_vects, xlat_vecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape[::-1], None, None)

    if ret:
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, img_shape[::-1], 1 ,img_shape[::-1])

        return LensCalibration(camera_matrix, dist_coeffs, new_camera_mtx, roi)
    else:
        raise CalibrationFailedError