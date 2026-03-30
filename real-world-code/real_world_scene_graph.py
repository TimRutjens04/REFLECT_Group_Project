import numpy as np
import math
import matplotlib.pyplot as plt
from transforms3d import euler
from scipy.spatial.transform import Rotation as R
from constants import *
from main.clip_utils import *
from main.point_cloud_utils import *
from main.utils import *
import os
import pickle

state_dict = {
    "fridge": ["fridge is open with white interior", "fridge is closed with gray door"],
    "black fridge": ["a fridge that has the fridge door open", "a fridge that has the fridge door closed"],
    "faucet": ["a faucet with blue light at the tip", "a faucet with no blue light at the tip"],
    "pot": ["a pot containing a cloth", "a pot containing nothing"], 
    "mug": ["a mug filled with black coffee", "a mug filled with water", "a mug that is empty"],
    "pressure cooker": ["a pressure cooker with lid", "a pressure cooker without lid"],
    "carrot": ["a small piece of carrot", "a full length carrot"],
    "orange carrot": ["a sliced carrot", "an unsliced carrot"],
    "purple beet": ["a purple beet that is sliced", "a purple beet that is not sliced"],
    "drawer": ["a drawer that is open with objects inside", "a drawer that is open with no object inside", "a drawer that is closed"],
    "yellow drawer": ["a yellow drawer that is open", "a yellow drawer that is closed"],
    "microwave": ['a microwave with its door open', 'a microwave with its door closed'],
    "microwave_closed": ['a microwave with an orange light', 'a microwave with no light']
}

# ref: https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py
def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1. get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin, 0.)
    ih = np.maximum(iymax-iymin, 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1]) +
           (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou, inters

def get_node_dist(pts_A, pts_B):
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(pts_A)
    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(pts_B)

    dists = pcd_A.compute_point_cloud_distance(pcd_B)
    dist = np.min(np.array(dists))
    return dist

def get_object_state(node_name, img):
    object_name = node_name.split("-")[0]
    if object_name in state_dict:
        states = state_dict[object_name]
        # rank based on current image
        img_feats = get_img_feats(img)
        state_feats = get_text_feats(states)
        sorted_states, sorted_scores = get_nn_text(states, state_feats, img_feats)
        print("obj states for: ", node_name)
        print("sorted_states: ", sorted_states)
        print("sorted_scores: ", sorted_scores)
        obj_state = sorted_states[0] 
        if obj_state in real_world_obj_state_map:
            obj_state = real_world_obj_state_map[obj_state]
        # Only need to check toggled on/off for microwave if door is closed
        if 'microwave' == object_name and 'with door closed' == obj_state:
            obj_state_additional = get_object_state('microwave_closed', img)
            obj_state = obj_state + ' and ' + obj_state_additional
        return obj_state
    return None


class Node(object):
    def __init__(self, name, object_id=None, pos3d=None, corner_pts=None, bbox2d=None, pcd=None, depth=None, global_node=False):
        self.name = name # bowl
        self.object_id = object_id # ai2thor object_id of bowl
        self.bbox2d = bbox2d # 2d bounding box (4x1)
        self.pos3d = pos3d # just position (no orientation) of the object (3x1)
        self.corner_pts = corner_pts # corner points of 3d bbox (8x3)
        self.pcd = pcd # point cloud (px3)
        self.depth = depth
        self.name_w_state = None
        self.global_node = global_node
        self.state = None

    def set_state(self, state):
        self.name_w_state = f"{self.name} ({state})"
        self.state = state

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return True if self.name == other.name else False

    def get_name(self):
        if self.name_w_state is not None:
            return self.name_w_state
        else:
            return self.name


class Edge(object):
    def __init__(self, start_node, end_node, edge_type="none"):
        self.start = start_node
        self.end = end_node
        self.edge_type = edge_type
    
    def __hash__(self):
        return hash((self.start, self.end, self.edge_type))

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end and self.edge_type == other.edge_type:
            return True
        else:
            return False

    def __str__(self):
        return str(self.start) + "->" + self.edge_type + "->" + str(self.end)


class SceneGraph(object):
    """
    Create a spatial scene graph
    """
    def __init__(self):
        self.nodes = []
        self.total_nodes = []
        self.edges = {}

    def add_node_wo_edge(self, node):
        self.total_nodes.append(node)

    def add_node(self, new_node, rgb=None):
        # add edges for this new_node (i.e. include this node in the scene graph)
        for node in self.total_nodes:
            if node.name != new_node.name:
                self.add_edge(node, new_node)
                self.add_edge(new_node, node)
        self.nodes.append(new_node)
        # Get object state of the new node
        new_node_name = new_node.name
        if new_node_name.split('-')[0] in state_dict:
            self.add_object_state(new_node, rgb, mode="clip")
        return new_node

    def add_edge(self, node, new_node):
        if "bowl" in new_node.name and "apple" in node.name:
            return
        pos_A, pos_B = node.pos3d, new_node.pos3d
        cam_arr = pos_B - pos_A
        norm_vector = cam_arr / np.linalg.norm(cam_arr)

        box_A, box_B = np.array(node.corner_pts), np.array(new_node.corner_pts)
        # print("box_A, box_B: ", box_A.shape, box_B.shape)
        if len(node.pcd) == 0 or len(new_node.pcd) == 0:
            return
        else:
            dist = get_pcd_dist(node.pcd, new_node.pcd)
            dist = dist/1000 # to convert into meters
        # if node.name == 'bowl' and new_node.name == 'lettuce slice':
        #     print(f"----------dist between {node.name} and {new_node.name}: {dist}")
        # if node.name == 'first stove burner':
        # print("-------------------- Adding edge -------------------")
        # print(f"dist between {node.name} and {new_node.name}: {dist}")
        
        box_A_pts, box_B_pts = np.array(node.pcd), np.array(new_node.pcd)
        # if node.name == 'cabinet' and new_node.name == 'cup':
        #     print("box_A: ", box_A)
        #     print("box_B: ", box_B)
            # print("box_A_pts.shpae, box_B_pts.shape: ", box_A_pts.shape, box_B_pts.shape)
            # print("box_A: ", box_A)

        # IN CONTACT
        if dist < 0.05:
            if new_node.name not in BULKY_OBJECTS:
                # --------------------
                if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=0.5):  # makeCoffee3: 0.2
                    if "countertop" in node.name or "stove burner" in node.name or "table" in node.name: # address the "inside countertop" issue
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
                    else:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
                elif len(np.where((box_B_pts[:, 0] < box_A[4, 0]) & (box_B_pts[:, 0] > box_A[0, 0]) & 
                        (box_B_pts[:, 2] < box_A[4, 2]) & (box_B_pts[:, 2] > box_A[0, 2]))[0]) > len(box_B_pts) * 0.7:
                    print(f">>> checking {new_node.name} on top of {node.name}")
                    if len(np.where(box_B_pts[:, 1] < box_A[4, 1])[0]) > len(box_B_pts) * 0.7:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
                    elif len(np.where(box_A_pts[:, 1] < box_B[4, 1])[0]) > len(box_A_pts) * 0.7:
                        self.edges[(node.name, new_node.name)] = Edge(node, new_node, "on top of")

        # CLOSE TO
        # https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20
        if dist < 0.4 and (new_node.name, node.name) not in self.edges and (not new_node.global_node):
            if node.name not in BULKY_OBJECTS and new_node.name not in BULKY_OBJECTS:
            # if True:
                if abs(norm_vector[1]) > 0.9:
                    if norm_vector[1] > 0:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "below")
                    else:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "above")
                elif abs(norm_vector[0]) > 0.8:
                    if norm_vector[0] > 0:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the right of")
                    else:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the left of")
                elif abs(norm_vector[2]) > 0.9 and new_node.bbox2d is not None and node.bbox2d is not None and new_node.depth is not None and node.depth is not None:
                    iou, inters = get_iou(new_node.bbox2d, node.bbox2d)
                    occlude_ratio = inters / ((node.bbox2d[2]-node.bbox2d[0]) * (node.bbox2d[3]-node.bbox2d[1]))
                    if occlude_ratio > 0.02 and len(np.where(new_node.depth <= np.min(node.depth))[0]) > len(new_node.depth) * 0.9:
                        self.edges[(new_node.name, node.name)] = Edge(new_node, node, "occluding")
                elif dist < 0.02:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "near")
    
    def add_object_state(self, node, rgb, mode="clip"):
        # remove later
        if rgb is not None:
            box = node.bbox2d
            # print("rgb: ", rgb.shape)
            h, w, _ = rgb.shape
            # print("in obj state box: ", box, h, w)
            h1 = max(0, int(box[1]-20))
            h2 = min(h-1, int(box[3]+20))
            w1 = max(0, int(box[0]-20))
            w2 = min(w-1, int(box[2]+20))
            cropped_img = rgb[h1:h2, w1:w2]
            
        if mode == "clip":
            state = get_object_state(node.name, cropped_img)
        if state is not None:
            node.set_state(state)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(rgb)
        # ax[1].imshow(cropped_img)
        # plt.show()
        return node

    def add_agent(self, gripper_pos, step_idx, args):
        obj_det_path = f'real_world/state_summary/{args.folder_name}/{args.obj_det}_obj_det/clip_processed_det'
        local_sg_path = f'real_world/state_summary/{args.folder_name}/local_graphs'
        # an object is in the grippers
        print(f"gripper_pos at {step_idx}: ", gripper_pos)
        # emperically tested this value
        if gripper_pos > 5:
            f_names = os.listdir(obj_det_path)
            sorted_f_names = sorted(f_names, key=lambda d: int(d.split('.')[0]))
            curr_obj_det_file_idx = sorted_f_names.index(f'{step_idx}.pickle')
            print("curr_obj_det_file_idx:", curr_obj_det_file_idx)
            # for all obj detection in current and surrounding frames
            for i in range(curr_obj_det_file_idx, max(-1,curr_obj_det_file_idx-2), -1):
                # if a detected obj is closer to the robot than a thresh
                file_idx = sorted_f_names[i].split('.')[0]
                print("file_idx: ", file_idx)
                # as a fix for when I run scene graph generation only for key frames -> Don't fret about this
                if int(file_idx) < step_idx - 500 or int(file_idx) > step_idx + 500:
                    continue
                
                if os.path.exists(f'{obj_det_path}/{file_idx}.pickle'):
                    with open(f'{obj_det_path}/{file_idx}.pickle', 'rb') as f:
                        obj_dets = pickle.load(f)
                # current frame
                if int(file_idx) == step_idx:
                    local_sg_nodes = self.nodes
                # past frames
                elif os.path.exists(f'{local_sg_path}/local_sg_{file_idx}.pkl'):
                    with open(f'{local_sg_path}/local_sg_{file_idx}.pkl', 'rb') as f:
                        local_sg = pickle.load(f)
                        local_sg_nodes = local_sg.nodes
                # future frames (not checking rn)
                else:
                    local_sg_nodes = []
                nodes_in_gripper = []
                for obj_name in obj_dets['labels']:
                    print("[DEBUG INFO]:", [o.name for o in local_sg_nodes])
                    print("[DEBUG INFO]:", obj_name)
                    nodes = [o for o in local_sg_nodes if o.name.split('-')[0] == obj_name]
                    for node in nodes:
                        print("label, step, center, dist: ", node.name, file_idx, node.pos3d, np.linalg.norm(node.pos3d/1000))
                        if node is not None and np.linalg.norm(node.pos3d/1000) < 0.9:
                            if node.name not in BULKY_OBJECTS:
                                self.edges[(node.name, "robot gripper")] = Edge(node, Node("robot gripper"), "inside")
                                nodes_in_gripper.append(node)
                if len(nodes_in_gripper) > 0:
                    return nodes_in_gripper
            self.edges[("nothing", "robot gripper")] = Edge(Node("nothing"), Node("robot gripper"), "inside")
        else:
            self.edges[("nothing", "robot gripper")] = Edge(Node("nothing"), Node("robot gripper"), "inside")
        return None

    def __eq__(self, other):
        if (set(self.nodes) == set(other.nodes)) and (set(self.edges.values()) == set(other.edges.values())):
            return True
        else:
            return False

    def visualize_graph(self, idx, np_image, bbox2d_dict):
        plt.figure(figsize=(16,10))
        ax = plt.gca()
        colors = COLORS * 255
        for (xmin, ymin, xmax, ymax), label, c in zip(list(bbox2d_dict.values()), list(bbox2d_dict.keys()), colors):
            ax.add_patch(plt.Circle(xy=((xmax + xmin)/2, (ymax + ymin)/2), radius=25, fill=True, alpha=0.5, color=c, linewidth=1))
            ax.text((xmax + xmin)/2, (ymax + ymin)/2, label, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        for edge in self.edges.values():
            if edge.edge_type != "none" and edge.start.bbox2d != None and edge.end.bbox2d != None:
                sx, sy = (edge.start.bbox2d[0] + edge.start.bbox2d[2])/2, (edge.start.bbox2d[1] + edge.start.bbox2d[3])/2
                ex, ey = (edge.end.bbox2d[0] + edge.end.bbox2d[2])/2, (edge.end.bbox2d[1] + edge.end.bbox2d[3])/2
                plt.arrow(sx, sy, (ex-sx), (ey-sy), width=2, head_width=15, head_length=10)
                ax.text(sx+(ex-sx)/4, sy+(ey-sy)/4, edge.edge_type, fontsize=15, color='yellow')

        plt.imshow(np_image)
        plt.axis('off')
        plt.savefig("images/graphs/frame_{}.png".format(idx))
        plt.close()

    def __str__(self):
        visited = []
        res = "[Nodes]:\n"
        for node in self.nodes:
            res += node.get_name()
            res += "\n"
        res += "\n"
        res += "[Edges]:\n"
        for edge_key, edge in self.edges.items():
            # print("edge_key, edge: ", edge_key, edge.edge_type)
            name_1, name_2 = edge_key
            edge_key_2 = (name_2, name_1)
            if (edge_key not in visited and edge_key_2 not in visited) or edge.edge_type == 'on top of' or edge.edge_type == 'inside':
                if edge.edge_type != "none":
                    res += str(edge)
                    res += "\n"
            visited.append(edge_key)
        return res
