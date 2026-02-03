import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
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
            }
        }

    def callback(self, image_msg, det_msg):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        oriented_array = OrientedComponentArray()
        oriented_array.header = image_msg.header
        
        for det in det_msg.detections:
            if det.class_name not in self.reference_data:
                continue


            self.get_logger().info(f'Processing component: {det.class_name}')

            # Extract bounding box
            x1, y1, x2, y2 = det.bbox
            crop = cv_image[y1:y2, x1:x2]
            
            ref_info = self.reference_data[det.class_name]
            ref_img = ref_info['image']
            
            # Find orientation using estimateAffinePartial2D
            # This typically involves feature matching (ORB/SIFT)
            # For simplicity, let's assume we have a function find_affine_transform
            matrix = self.find_affine_transform(ref_img, crop)
            
            if matrix is not None:
                comp = OrientedComponent()
                comp.class_name = det.class_name
                
                # Extract rotation and translation from the matrix
                # matrix = [[s*cos(th), -s*sin(th), tx], [s*sin(th), s*cos(th), ty]]
                tx = matrix[0, 2]
                ty = matrix[1, 2]
                angle = np.arctan2(matrix[1, 0], matrix[0, 0])

                self.get_logger().info(f'Rotation: {angle}')
                
                comp.center = Point(x=float(x1 + tx), y=float(y1 + ty), z=0.0)
                comp.rotation = float(angle)
                
                # Transform connection points
                for name, (rx, ry) in ref_info['connections'].items():
                    # Apply rotation and translation
                    # [nx, ny] = [[cos, -sin], [sin, cos]] * [rx, ry] + [tx, ty]
                    nx = rx * np.cos(angle) - ry * np.sin(angle) + tx + x1
                    ny = rx * np.sin(angle) + ry * np.cos(angle) + ty + y1
                    
                    conn_point = ConnectionPoint()
                    conn_point.name = name
                    conn_point.x = float(nx)
                    conn_point.y = float(ny)
                    comp.connection_points.append(conn_point)
                
                oriented_array.components.append(comp)
        
        self.publisher_.publish(oriented_array)

    def find_affine_transform(self, ref_img, target_img):
        # Feature-based matching to find the affine transform
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(ref_img, None)
        kp2, des2 = orb.detectAndCompute(target_img, None)
        
        if des1 is None or des2 is None:
            return None
            
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        self.get_logger().info(f'Matches: {len(matches)}')
        
        if len(matches) < 4:
            return None
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        self.get_logger().info(f'Matrix: {matrix}')
        return matrix

def main(args=None):
    rclpy.init(args=args)
    node = OrientationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
