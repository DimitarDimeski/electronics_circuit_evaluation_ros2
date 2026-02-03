import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.timer_period = 10.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.bridge = CvBridge()
        self.image_path = '/circuit_detection.v5i.coco/train/Screenshot-from-2025-12-08-20-38-54_png.rf.d89225c00792aaa43dc1e665ceffe222.jpg'
        

    def timer_callback(self):
        if not os.path.exists(self.image_path):
            self.get_logger().error(f'Image file not found: {self.image_path}')
            return

        cv_image = cv2.imread(self.image_path)
        if cv_image is not None:
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_color_optical_frame'
            self.publisher_.publish(msg)
            self.get_logger().info(f'Successfully published image from {self.image_path}')
        else:
            self.get_logger().error(f'Failed to decode image from {self.image_path}')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
