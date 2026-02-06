import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os 
import supervision as sv
from inference import get_model
from rfdetr import RFDETRNano
from io import BytesIO
import requests
from electronics_circuit_evaluation_msgs.msg import Detection, DetectionArray

# Import RF-DETR dependencies here
# try:
#     from rf_detr import RFDETR
# except ImportError:
#     print("RF-DETR library not found. Please install it or provide the path.")

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10)
        self.detections_publisher_ = self.create_publisher(DetectionArray, '/electronics_circuit_evaluation/detections', 10)
        self.image_publisher_ = self.create_publisher(Image, '/electronics_circuit_evaluation/image', 10)
        self.bridge = CvBridge()
        
        # Load RF-DETR model
        self.get_logger().info('Initializing RF-DETR model...')
        self.model = RFDETRNano(pretrain_weights='/circuit_detection.v5i.coco/trained/030220251140/checkpoint_best_total.pth')


        color = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
            "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
        ])

        self.bbox_annotator = sv.BoxAnnotator(color=color)
        self.label_annotator = sv.LabelAnnotator(
            color=color,
            text_color=sv.Color.BLACK)

        # self.model = RFDETR(weights='path/to/weights.pth')
        self.get_logger().info('Detection node started.')

        self.class_names = ['components', 'battery_holder_module', 'connecting_plug', 'connector_angled', 'connector_angled_with_socket', 
                             'connector_interrupted', 'connector_straight', 'connector_straight_with_socket', 'connector_t_shaped', 'glass_tank',
                            'human_model_for_electrical_safety', 'junction',
                            'npn_transistor', 'ntc_resistor', 'ptc_resistor', 'resistor_100', 'resistor_10k',
                            'resistor_50', 'socket_for_incandescent_lamp', 'switch_change_over', 'switch_on_off']

    def listener_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Perform detection
        detections = self.model.predict(cv_image, confidence=0.8)

        detections_labels = [
        f"{self.class_names[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
        ]
        
        self.get_logger().info(f'Detections: {detections}')

        detections_image = cv_image.copy()
        detections_image = self.bbox_annotator.annotate(detections_image, detections)
        detections_image = self.label_annotator.annotate(detections_image, detections, detections_labels)
        
        detection_array = DetectionArray()
        detection_array.header = msg.header
        
        for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            msg_det = Detection()
            msg_det.class_name = self.class_names[class_id]
            msg_det.confidence = float(confidence)
            msg_det.bbox = [int(x) for x in xyxy]
            detection_array.detections.append(msg_det)
            
        self.detections_publisher_.publish(detection_array)
        self.image_publisher_.publish(self.bridge.cv2_to_imgmsg(detections_image, 'bgr8'))
        # self.get_logger().info(f'Published {len(detection_array.detections)} detections.')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
