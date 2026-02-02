import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os 
import supervision as sv
from inference import get_model
from PIL import Image
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
        self.model = get_model('circuit_detection-q6xm3')
        # self.model = RFDETR(weights='path/to/weights.pth')
        self.get_logger().info('Detection node started.')

    def listener_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Perform detection
        predictions = self.model.infer(cv_image, confidence=0.5)[0]

        detections = sv.Detections.from_inference(predictions)
       
        self.get_logger().info(f'Detections: {detections}')

        labels = [prediction.class_name for prediction in predictions.predictions]

        annotated_image = cv_image.copy()
        annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
        
        detection_array = DetectionArray()
        detection_array.header = msg.header
        
        for det in detections:
            msg_det = Detection()
            msg_det.class_name = det['class']
            msg_det.confidence = float(det['conf'])
            msg_det.bbox = [int(x) for x in det['bbox']]
            detection_array.detections.append(msg_det)
            
        self.publisher_.publish(detection_array)
        self.image_publisher_.publish(self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8'))
        # self.get_logger().info(f'Published {len(detection_array.detections)} detections.')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
