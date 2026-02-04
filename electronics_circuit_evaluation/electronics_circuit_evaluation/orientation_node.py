import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import math
from electronics_circuit_evaluation_msgs.msg import Detection, DetectionArray, OrientedComponent, OrientedComponentArray, ConnectionPoint
from geometry_msgs.msg import Point
import message_filters

class OrientationNode(Node):
    def __init__(self):
        super().__init__('orientation_node')
        
        # Use message_filters to synchronize Image and Detections
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.det_sub = message_filters.Subscriber(self, DetectionArray, '/electronics_circuit_evaluation/detections')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.det_sub], 10, 0.05)
        self.ts.registerCallback(self.callback)
        
        self.publisher_ = self.create_publisher(OrientedComponentArray, 'oriented_components', 10)
        self.debug_image_pub = self.create_publisher(Image, '/electronics_circuit_evaluation/orientation_debug', 10)
        self.bridge = CvBridge()
        
        # Reference mapping: class_name -> {image: np.array, connections: {name: (x, y)}}
        # Coordinates are relative to the center of the reference image
        self.reference_data = {}
        self.load_references()
        
        self.get_logger().info('Orientation node started.')

    def load_references(self):
        # Placeholder for loading reference images and connection points
        # Example:
        # self.reference_data['resistor'] = {
        #     'image': cv2.imread('path/to/resistor_ref.png'),
        #     'connections': {'P1': (-50, 0), 'P2': (50, 0)}
        # }

        self.reference_data= {
            'connector_angled': {
                'image': cv2.imread('/circuit_detection.v5i.coco/cropped_ref_images/connector_angled.png'),
                'connections': {'P1': (0, 35), 'P2': (25, 0)}
            },
            'switch_on_off': {
                'image': cv2.imread('/circuit_detection.v5i.coco/cropped_ref_images/switch_on_off.png'),
                'connections': {'P1': (-25, 0), 'P2': (25, 0)}
            },
            'connector_straight': {
                'image': cv2.imread('/circuit_detection.v5i.coco/cropped_ref_images/connector_straight.png'),
                'connections': {'P1': (-25, 0), 'P2': (25, 0)}
            },
            'connector_t_shaped': {
                'image': cv2.imread('/circuit_detection.v5i.coco/cropped_ref_images/connector_t_shaped.png'),
                'connections': {'P1': (-25, 0), 'P2': (25, 0), 'P3': (0, 35)}
            },
            'socket_for_incandescent_lamp': {
                'image': cv2.imread('/circuit_detection.v5i.coco/cropped_ref_images/socket_for_incandescent_lamp.png'),
                'connections': {'P1': (-25, 0), 'P2': (25, 0)}
            },
            'junction': {
                'image': cv2.imread('/circuit_detection.v5i.coco/cropped_ref_images/junction.png'),
                'connections': {'P1': (-25, 0)}
            },
            'connector_interrupted': {
                'image': cv2.imread('/circuit_detection.v5i.coco/cropped_ref_images/connector_interrupted.png'),
                'connections': {'P1': (-25, 0), 'P2': (25, 0)}
            }
        }

    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def rotate_mask(self, mask, angle_deg):
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    def callback(self, image_msg, det_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        oriented_array = OrientedComponentArray()
        oriented_array.header = image_msg.header
        
        debug_images = []
        
        for det in det_msg.detections:
            if det.class_name not in self.reference_data:
                continue

            self.get_logger().info(f'Processing component: {det.class_name}')

            # Extract bounding box
            x1, y1, x2, y2 = det.bbox
            crop = cv_image[y1:y2, x1:x2]
            
            ref_info = self.reference_data[det.class_name]
            ref_img_color = ref_info['image'].copy()

            # Store original dimensions for coordinate mapping
            orig_h, orig_w = crop.shape[:2]
            ref_h_orig, ref_w_orig = ref_img_color.shape[:2]

            # --- 1. Resize both to 50x50 ---
            crop = cv2.resize(crop, (50, 50))
            ref_img_color = cv2.resize(ref_img_color, (50, 50))
            crop_color = crop.copy()

            # --- 2. Apply circular crop (radius 25 from center) ---
            def circular_mask(img, radius=20):
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
                return cv2.bitwise_and(img, img, mask=mask)

            ref_img_color = circular_mask(ref_img_color)
            crop_color = circular_mask(crop_color)

            # --- 3. Convert to HSV and threshold white ---
            hsv_ref = cv2.cvtColor(ref_img_color, cv2.COLOR_BGR2HSV)

            # Broadened white range
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 120, 255])
            mask_ref = cv2.inRange(hsv_ref, lower_white, upper_white)

            # --- 4. Morphological cleanup ---
            kernel_close = np.ones((5, 5), np.uint8)
            kernel_open = np.ones((3, 3), np.uint8)
            
            #mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, kernel_close)
            #mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_OPEN, kernel_open)

            hsv_crop = cv2.cvtColor(crop_color, cv2.COLOR_BGR2HSV)
            mask_crop = cv2.inRange(hsv_crop, lower_white, upper_white)
            #mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel_close)
            #mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel_open)

            ref_img_bin = mask_ref
            crop_bin = mask_crop

            if ref_img_bin is None:
                self.get_logger().error(f'Reference image for {det.class_name} not found.')
                continue

            self.get_logger().info(f'Reference image: {ref_img_bin.shape}')
            self.get_logger().info(f'Crop image: {crop_bin.shape}')
            
            angle1, _ = self.orientation_from_white_symbol(crop_bin)
            angle2, _ = self.orientation_from_white_symbol(ref_img_bin)

            # The PCA angle is the orientation of the principal axis in [-90, 90)
            # We want to find the rotation that maps ref to crop.
            # Base rotation candidate:
            base_angle = angle1 - angle2
            
            # Since PCA has 180-degree ambiguity, we check both theta and theta + 180
            candidates = [base_angle, base_angle + 180]
            
            best_iou = -1
            best_angle_snapped = 0
            candidate_masks = []
            
            for cand in candidates:
                # Round each candidate to the nearest 90 degrees
                cand_snapped = round(cand / 90.0) * 90.0
                
                # Rotate reference mask to see if it matches the crop mask
                rotated_ref = self.rotate_mask(ref_img_bin, cand_snapped)
                candidate_masks.append(cv2.cvtColor(rotated_ref, cv2.COLOR_GRAY2BGR))
                
                iou = self.calculate_iou(rotated_ref, crop_bin)
                
                if iou > best_iou:
                    best_iou = iou
                    best_angle_snapped = cand_snapped
            
            relative_rotation = best_angle_snapped
            self.get_logger().info(f'Disambiguated relative rotation (deg): {relative_rotation:.2f} (IoU: {best_iou:.4f})')

            # Create side-by-side image for debug: [Ref Color | Ref Mask | Rot 1 | Rot 2 | Crop Mask | Crop Color]
            mask_ref_bgr = cv2.cvtColor(ref_img_bin, cv2.COLOR_GRAY2BGR)
            mask_crop_bgr = cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2BGR)

            combined = np.hstack((ref_img_color, mask_ref_bgr, candidate_masks[0], candidate_masks[1], mask_crop_bgr, crop_color))
            debug_images.append(combined)

            # Build matrix using the PCA-based snapped rotation
            matrix = cv2.getRotationMatrix2D((25, 25), relative_rotation, 1.0)
            
            if matrix is not None:
                comp = OrientedComponent()
                comp.class_name = det.class_name
                
                # In 50x50 space, center is 25,25
                ref_center_x, ref_center_y = 25.0, 25.0

                # Transform center point in 50x50 space
                center_in_crop_50 = matrix @ np.array([ref_center_x, ref_center_y, 1.0])
                
                # Scale factors to map back to original image coordinates
                scale_x = orig_w / 50.0
                scale_y = orig_h / 50.0

                # Global coordinates
                comp.center = Point(x=float(x1 + center_in_crop_50[0] * scale_x), 
                                  y=float(y1 + center_in_crop_50[1] * scale_y), 
                                  z=0.0)
                
                # The rotation is already snapped to 90 degrees
                comp.rotation = float(np.radians(relative_rotation))
                
                self.get_logger().info(f'Final snapped rotation: {relative_rotation:.2f} degrees')
                
                # Transform connection points
                for name, (rx, ry) in ref_info['connections'].items():
                    # rx, ry are relative to original center. Scale to 50x50 space:
                    rx_50 = rx * (50.0 / ref_w_orig)
                    ry_50 = ry * (50.0 / ref_h_orig)
                    
                    abs_ref_x_50 = 25.0 + rx_50
                    abs_ref_y_50 = 25.0 + ry_50
                    
                    # Transform to crop coordinates in 50x50 space
                    conn_in_crop_50 = matrix @ np.array([abs_ref_x_50, abs_ref_y_50, 1.0])
                    
                    conn_point = ConnectionPoint()
                    conn_point.name = name
                    conn_point.x = float(x1 + conn_in_crop_50[0] * scale_x)
                    conn_point.y = float(y1 + conn_in_crop_50[1] * scale_y)
                    comp.connection_points.append(conn_point)
                
                oriented_array.components.append(comp)
        
        self.publisher_.publish(oriented_array)

        if debug_images:
            # Stack all debug images vertically
            max_w = max(img.shape[1] for img in debug_images)
            processed_debug_images = []
            for img in debug_images:
                # Ensure 3 channels for visualization (masks are 1-channel)
                if len(img.shape) == 2:
                    img_viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_viz = img
                
                h, w = img_viz.shape[:2]
                if w < max_w:
                    padded = np.zeros((h, max_w, 3), dtype=np.uint8)
                    padded[:, :w, :] = img_viz
                    processed_debug_images.append(padded)
                else:
                    processed_debug_images.append(img_viz)
            
            final_debug_image = np.vstack(processed_debug_images)
            debug_msg = self.bridge.cv2_to_imgmsg(final_debug_image, encoding='bgr8')
            debug_msg.header = image_msg.header
            self.debug_image_pub.publish(debug_msg)


    def orientation_from_white_symbol(
        self,
        image_bgr,
        min_area=10
    ):
        """
        Estimate orientation (in degrees) of a white symbol on a blue puck
        using contour + PCA.

        Args:
            image_bgr (np.ndarray): Input image (BGR, OpenCV format)
            min_area (int): Minimum contour area to keep

        Returns:
            angle_deg (float): Orientation angle in degrees, range [-90, 90)
            debug (dict): Optional debug outputs
        """

        mask = image_bgr

        # --- 3. Find contours ---
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        self.get_logger().info(f'Contours: {len(contours)}')

        # Keep only meaningful contours
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if len(contours) == 0:
            return 0, None

        # Merge all contours into one point cloud
        pts = np.vstack(contours).squeeze().astype(np.float32)

        # --- 4. PCA ---
        mean = np.mean(pts, axis=0)
        pts_centered = pts - mean

        cov = np.cov(pts_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Dominant direction
        idx = np.argmax(eigenvalues)
        direction = eigenvectors[:, idx]

        # --- 5. Angle computation ---
        angle_rad = math.atan2(direction[1], direction[0])
        angle_deg = np.degrees(angle_rad)

        # Normalize to [-90, 90)
        if angle_deg < -90:
            angle_deg += 180
        elif angle_deg >= 90:
            angle_deg -= 180

        debug = {
            "mask": mask,
            "points": pts,
            "mean": mean,
            "direction": direction
        }

        return angle_deg, debug


def main(args=None):
    rclpy.init(args=args)
    node = OrientationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
