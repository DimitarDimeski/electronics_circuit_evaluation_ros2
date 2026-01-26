import rclpy
from rclpy.node import Node
import json
from electronics_circuit_evaluation_msgs.msg import OrientedComponent, OrientedComponentArray, Netlist
import numpy as np

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

class ConnectivityNode(Node):
    def __init__(self):
        super().__init__('connectivity_node')
        self.subscription = self.create_subscription(
            OrientedComponentArray,
            'oriented_components',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Netlist, '/electronics_circuit_evaluation/netlist', 10)
        
        # Proximity threshold in pixels
        self.threshold = 15.0
        self.get_logger().info('Connectivity node started.')

    def listener_callback(self, msg):
        all_points = []
        # Store points as (component_index, point_index, x, y)
        for i, comp in enumerate(msg.components):
            for j, pt in enumerate(comp.connection_points):
                all_points.append({
                    'comp_idx': i,
                    'pt_idx': j,
                    'x': pt.x,
                    'y': pt.y,
                    'comp_name': f"{comp.class_name}_{i}",
                    'pt_name': pt.name
                })
        
        if not all_points:
            return
            
        n = len(all_points)
        dsu = DSU(n)
        
        # Proximity search
        for i in range(n):
            for j in range(i + 1, n):
                p1 = all_points[i]
                p2 = all_points[j]
                dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                
                if dist < self.threshold:
                    dsu.union(i, j)
        
        # Group into nets
        nets = {}
        for i in range(n):
            root = dsu.find(i)
            if root not in nets:
                nets[root] = []
            nets[root].append(all_points[i])
            
        # Format netlist
        netlist_data = {
            'components': [],
            'nets': []
        }
        
        for i, comp in enumerate(msg.components):
            comp_info = {
                'id': f"{comp.class_name}_{i}",
                'type': comp.class_name,
                'connections': {}
            }
            for j, pt in enumerate(comp.connection_points):
                # Find which net this point belongs to
                point_idx = next(idx for idx, p in enumerate(all_points) if p['comp_idx'] == i and p['pt_idx'] == j)
                net_id = dsu.find(point_idx)
                comp_info['connections'][pt.name] = f"net_{net_id}"
            netlist_data['components'].append(comp_info)
            
        for net_id, points in nets.items():
            netlist_data['nets'].append({
                'id': f"net_{net_id}",
                'points': [f"{p['comp_name']}.{p['pt_name']}" for p in points]
            })
            
        netlist_msg = Netlist()
        netlist_msg.header = msg.header
        netlist_msg.netlist_json = json.dumps(netlist_data)
        self.publisher_.publish(netlist_msg)
        # self.get_logger().info('Published netlist.')

def main(args=None):
    rclpy.init(args=args)
    node = ConnectivityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
