import rclpy
from rclpy.node import Node
import json
from electronics_circuit_evaluation_msgs.msg import Netlist, EvaluationResult
try:
    from skidl import Part, Net, Circuit, ERC, lib_search_paths, FOOTPRINT
except ImportError:
    print("skidl library not found. Please install it with 'pip install skidl'.")

class EvaluationNode(Node):
    def __init__(self):
        super().__init__('evaluation_node')
        self.subscription = self.create_subscription(
            Netlist,
            'netlist',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(EvaluationResult, 'evaluation_result', 10)
        
        # Mapping component types to skidl-compatible Parts
        # This requires skidl libraries to be installed and accessible
        self.type_to_part = {
            'resistor': 'R',
            'battery': 'V',
            'led': 'D',
            'capacitor': 'C',
            'switch': 'SW_Push'
        }
        
        self.get_logger().info('Evaluation node started.')

    def listener_callback(self, msg):
        try:
            netlist_data = json.loads(msg.netlist_json)
        except json.JSONDecodeError:
            self.get_logger().error('Failed to decode netlist JSON.')
            return
            
        eval_result = EvaluationResult()
        eval_result.header = msg.header
        
        try:
            # Clear previous circuit state in skidl
            from skidl import default_circuit
            default_circuit.reset()
            
            nets = {}
            # Pre-create all nets
            for net_info in netlist_data['nets']:
                nets[net_info['id']] = Net(net_info['id'])
            
            # Instantiate components and connect them
            for comp_info in netlist_data['components']:
                comp_type = comp_info['type'].lower()
                if comp_type not in self.type_to_part:
                    eval_result.warnings.append(f"Unknown component type: {comp_type}. Skipping ERC for this component.")
                    continue
                
                # Create skidl Part
                # Note: This assumes standard libraries like 'Device' are available
                try:
                    part_name = self.type_to_part[comp_type]
                    # In a real scenario, you might need to specify the library, e.g., Part('Device', 'R')
                    # For this example, we assume Part(part_name) works or the user has set up skidl libraries.
                    p = Part('Device', part_name, dest=Circuit.CONTEXT)
                    
                    # Map netlist pins to skidl pins
                    # This mapping depends on how the puck connection points are named
                    # e.g., 'P1' -> pin 1, 'P2' -> pin 2
                    for conn_name, net_id in comp_info['connections'].items():
                        if net_id in nets:
                            # Try to find pin by name or index
                            pin = p[conn_name] if conn_name in p else None
                            if pin:
                                nets[net_id] += pin
                            else:
                                # Fallback to numeric mapping if conn_name is 'P1', 'P2' etc.
                                if conn_name.startswith('P') and conn_name[1:].isdigit():
                                    pin_idx = int(conn_name[1:])
                                    p[pin_idx] += nets[net_id]
                except Exception as e:
                    eval_result.errors.append(f"Failed to instantiate {comp_info['id']}: {str(e)}")

            # Run skidl ERC
            # Capturing ERC output might require redirecting stdout/stderr or using skidl's logger
            # For simplicity, we just call ERC() and catch exceptions
            try:
                # ERC() in skidl often prints to stdout. 
                # To be more robust, we could check for floating nets or shorts manually using skidl's API
                ERC()
                
                # Check for common issues manually as well
                for net_id, net in nets.items():
                    if len(net) < 2:
                        eval_result.warnings.append(f"Net {net_id} has only {len(net)} connections (possible floating pin).")
                
                eval_result.success = True
                eval_result.message = "ERC completed successfully."
            except Exception as e:
                eval_result.success = False
                eval_result.errors.append(f"ERC failed: {str(e)}")
                
        except Exception as e:
            eval_result.success = False
            eval_result.errors.append(f"Internal error during evaluation: {str(e)}")
            
        self.publisher_.publish(eval_result)
        # self.get_logger().info('Published evaluation result.')

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
