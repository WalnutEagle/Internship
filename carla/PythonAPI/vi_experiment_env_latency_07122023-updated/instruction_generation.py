from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
import signal
import argparse
from skimage import measure
from freemap import *
from freemap_extraction import Tags
from astar import AStar


# random.seed(2022)
EPS = 10e-4
MAX_FAIL_ATTEMPT = 10


class InstructionGeneration():
    ACTION_SET = {'walk': ['walk straight', 'go straight', 'go forward', 'walk forward'],
                  'stop': ['stop'], 'turn': ['turn'], 'indication': ['to your']}
    DIRECTION = {1: "left", -1: "right"}
    ACTION_AMOUNT = {"little": ["a little", "a bit", "slightly"]}
    TURN_CLOCKFACE = {-30: "one", -60: "two", -90: "three", 30: "eleven", 60: "ten", 90: "nine"}

    def __init__(self, initial_point, path, semantics_img, m_per_pixel):
        # point = (x, y, yaw)
        self.init_pt = initial_point
        self.path = path
        self.yaw = self.get_yaw()
        self.m_per_pixel = m_per_pixel
        self.semantics_img = semantics_img
        return

    def get_yaw(self):
        yaw = np.zeros(len(self.path))
        yaw[0] = self.init_pt[2]
        diff = np.diff(self.path, axis=0)
        rad = np.arctan2(diff[:, 1], diff[:, 0])
        yaw[1:] = rad + 3*np.pi/2
        # offset -pi/2 making 0 axis pointing forward
        # yaw = yaw - np.pi/2
        return yaw

    def get_dyaw(self):
        dyaw = np.diff(self.yaw)
        return dyaw

    def get_instructions(self):
        instructions = []
        clusters = []
        sub_instruction = []
        sub_clusters = []
        poc, poc_dyaw, true_poc = self.path_process_pipeline()
        poc_dyaw = rad2deg(poc_dyaw)
        # print(poc)
        next_idx = 0
        idx = 0
        while idx < len(poc)-1:
            next_idx = idx + 1
            next_p = poc[next_idx]
            p = poc[idx]
            next_coord = self.path[next_p]
            curr_coord = self.path[p]
            distance = np.linalg.norm((curr_coord - next_coord)*self.m_per_pixel)
            dyaw = round(poc_dyaw[idx])
            if abs(dyaw) == 180:
                sub_instruction += [random.choice(self.ACTION_SET['turn'])]
                sub_instruction += ['around']
                sub_clusters += ['turn', 'around']
            elif abs(dyaw) >= 30:
                sub_instruction += [random.choice(self.ACTION_SET['turn'])]
                sub_instruction += [random.choice(self.ACTION_SET['indication'])]
                sub_instruction += [self.TURN_CLOCKFACE[dyaw]]
                sub_clusters += ['turn']
                sub_clusters += ['left' if dyaw > 0 else 'right']
            elif EPS < abs(dyaw) < 30:
                sub_instruction += [random.choice(self.ACTION_SET['turn'])]
                sub_instruction += [self.DIRECTION[np.sign(dyaw)]]
                sub_instruction += [random.choice(self.ACTION_AMOUNT['little'])]
                sub_clusters += ['turn']
                sub_clusters += ['left' if dyaw > 0 else 'right']
                sub_clusters += ['slight']

            if distance >= 0.5:
                sub_instruction += [random.choice(self.ACTION_SET['walk'])]
                sub_instruction += [str(round2base(distance, 0.5)), 'meters']
                sub_clusters += ['walk']
            elif EPS < distance < 0.5:
                sub_instruction += [random.choice(self.ACTION_SET['walk'])]
                sub_instruction += [random.choice(self.ACTION_AMOUNT['little'])]
                sub_clusters += ['walk', 'slight']
            if distance > 0.5:
                instructions += [' '.join(sub_instruction)]
                clusters += [' '.join(sub_clusters)]
                sub_instruction = []
                sub_clusters = []
            idx += 1
        instructions += [random.choice(self.ACTION_SET['stop'])]
        return instructions, clusters, true_poc

    def get_description(self, true_poc, radius_in_m=2, thresh=0.1, max_obstacle_number=3):
        descriptions = []
        clusters = []
        yaws = self.get_yaw()
        obstacle_contours = self.get_obstacles()
        tag_hash_inv = list(Tags.Hash.keys())
        radius = radius_in_m//self.m_per_pixel
        # assume 90 fov
        viewable_mask = np.tril(np.ones(self.semantics_img.shape)) * \
            np.flip(np.tril(np.ones(self.semantics_img.shape)), axis=1)
        viewable_semantics = self.semantics_img * viewable_mask
        # add poc if start is mission
        description_poc = true_poc.copy()
        if 0 not in description_poc:
            description_poc = [0] + description_poc
        for p in description_poc:
            sub_descriptions = []
            sub_clusters = []
            left_obstacles = []
            right_obstacles = []
            front_obstacles = []
            right_descriptions = []
            left_descriptions = []
            front_descriptions = []
            yaw = wrap2pi(yaws[p])
            if np.abs(yaw) > np.pi/2:   # skip if yaw too large, out of view
                descriptions += ['']
                continue
            detected_obstacle = dict.fromkeys(obstacle_contours.keys())
            pos = self.path[p]
            neighbour_bbox = self.get_neighbour_bbox(pos, radius)
            left_thresh = neighbour_bbox[2][0]
            right_thresh = neighbour_bbox[1][0]
            front_thresh = neighbour_bbox[1][1]
            back_thresh = neighbour_bbox[2][1]
            regional_semantics = viewable_semantics[back_thresh: front_thresh, left_thresh: right_thresh]

            for tag in obstacle_contours:
                loc = np.where(regional_semantics == Tags.Hash[tag])
                if len(loc[0]) > 0:  # if loc not empty -> exist in region
                    for contour_id, contour in enumerate(obstacle_contours[tag]):
                        if any(np.less_equal(contour, [right_thresh, front_thresh]).all(1)
                                & np.greater_equal(contour, [left_thresh, back_thresh]).all(1)):
                            if detected_obstacle[tag] == None:
                                detected_obstacle[tag] = []
                            if contour_id not in detected_obstacle[tag]:
                                detected_obstacle[tag] += [contour_id]
            # find close object and assign direction
            for tag in detected_obstacle:
                obstacles = detected_obstacle[tag]
                if obstacles == None:
                    continue
                for obs_id in obstacles:
                    contour = obstacle_contours[tag][obs_id]
                    close_contour = contour[(contour[:, 0] >= left_thresh) & (contour[:, 0] <= right_thresh) &
                                            (contour[:, 1] <= front_thresh) & (contour[:, 0] >= left_thresh)]
                    closest = close_contour[np.argmin(np.linalg.norm(close_contour - pos, axis=1))]
                    diff = closest - pos
                    norm = np.linalg.norm(diff)
                    if diff[0] > 0 + thresh//self.m_per_pixel:
                        right_obstacles += [(tag, norm)]
                    elif diff[0] < 0 - thresh//self.m_per_pixel:
                        left_obstacles += [(tag, norm)]
                    else:
                        front_obstacles += [(tag, norm)]
            # generate description
            right_obstacles.sort(key=lambda tup: tup[1])
            left_obstacles.sort(key=lambda tup: tup[1])
            front_obstacles.sort(key=lambda tup: tup[1])
            for idx, obs_tup in enumerate(right_obstacles):
                if idx > max_obstacle_number:
                    break
                right_descriptions += [obs_tup[0]]
            for idx, obs_tup in enumerate(left_obstacles):
                if idx > max_obstacle_number:
                    break
                left_descriptions += [obs_tup[0]]
            for idx, obs_tup in enumerate(front_obstacles):
                if idx > max_obstacle_number:
                    break
                front_descriptions += [obs_tup[0]]
            if left_descriptions:
                filt_left_descriptions = self.filter_description(left_descriptions)
                left_descriptions_unique = list(set(filt_left_descriptions))
                sub_descriptions += [' and '.join(left_descriptions_unique)]
                sub_descriptions += ['on your left']
                sub_clusters += [' '.join(left_descriptions_unique) + ' left']
            if right_descriptions:
                filt_right_descriptions = self.filter_description(right_descriptions)
                right_descriptions_unique = list(set(filt_right_descriptions))
                sub_descriptions += [' and '.join(list(set(right_descriptions_unique)))]
                sub_descriptions += ['on your right']
                sub_clusters += [' '.join(right_descriptions_unique) + ' right']
            if front_descriptions:
                filt_front_descriptions = self.filter_description(front_descriptions)
                front_descriptions_unique = list(set(filt_front_descriptions))
                sub_descriptions += [' and '.join(front_descriptions)]
                sub_descriptions += ['in front']
                sub_clusters += [' '.join(front_descriptions_unique) + ' front']
            descriptions += [' '.join(sub_descriptions)]
            clusters += [' '.join(sub_clusters)]
        return descriptions, clusters, description_poc

    def filter_description(self, description_list):
        filtered_descriptions = ['PEDESTRIAN' if (d == 'VISUALLYIMPAIRED' or d ==
                                    'WHEELPD') else d for d in description_list]
        return filtered_descriptions

    def get_neighbour_bbox(self, center, radius):
        x = center[0]
        y = center[1]
        lu = [int(x-radius), int(y+2*radius)]
        ru = [int(x+radius), int(y+2*radius)]
        ld = [int(x-radius), int(y)]
        rd = [int(x+radius), int(y)]
        return [lu, ru, ld, rd]

    def path_process_pipeline(self):
        dyaw = self.get_dyaw()
        dyaw = wrap2pi(dyaw)
        poc = np.where(dyaw != 0)[0]
        if 0 not in poc:
            poc = np.concatenate(([0], poc), axis=0)
        poc = np.concatenate((poc, [len(dyaw) - 1]))  # add the endpoint to poc
        poc_dyaw = dyaw[poc]
        # print(poc)
        poc, poc_dyaw = self.merge_short_segments(poc, poc_dyaw)
        true_poc = poc.copy()
        poc, poc_dyaw = self.process_large_turns(poc, poc_dyaw)
        # print('process_large_turns:', poc, poc_dyaw)
        poc, poc_dyaw = self.process_small_turns(poc, poc_dyaw)
        return poc, poc_dyaw, true_poc

    def merge_short_segments(self, poc, poc_dyaw, dist_thresh=0.5):
        # merge close consecutive turns
        new_poc = []
        new_poc_dyaw = []
        cur_idx = 0
        next_idx = cur_idx + 1
        accum_dyaw = poc_dyaw[cur_idx]
        while next_idx < len(poc):
            next_p = poc[next_idx]
            p = poc[cur_idx]
            next_coord = self.path[next_p]
            curr_coord = self.path[p]
            look_ahead_distance = np.linalg.norm((next_coord - curr_coord)*self.m_per_pixel)
            if look_ahead_distance < dist_thresh:
                accum_dyaw += poc_dyaw[next_idx]
                next_idx += 1
            else:
                new_poc += [p]
                new_poc_dyaw += [accum_dyaw]
                cur_idx = next_idx
                next_idx = cur_idx + 1
                accum_dyaw = poc_dyaw[cur_idx]
        new_poc += [poc[-1]]
        new_poc_dyaw += [poc_dyaw[-1]]
        return new_poc, new_poc_dyaw

    def process_large_turns(self, poc, poc_dyaw):
        # split dyaw thats more than 90 deg
        new_poc = []
        new_poc_dyaw = []
        for idx, p in enumerate(poc):
            if poc_dyaw[idx] >= 0:
                # left turns
                if poc_dyaw[idx] > np.pi/2:
                    res = poc_dyaw[idx] - np.pi
                    new_poc += [p, p]
                    new_poc_dyaw += [np.pi, res]
                else:
                    new_poc += [p]
                    new_poc_dyaw += [poc_dyaw[idx]]
            if poc_dyaw[idx] < 0:
                # right turns
                if poc_dyaw[idx] < -np.pi/2:
                    res = poc_dyaw[idx] + np.pi
                    new_poc += [p, p]
                    new_poc_dyaw += [np.pi, res]
                else:
                    new_poc += [p]
                    new_poc_dyaw += [poc_dyaw[idx]]
        return new_poc, new_poc_dyaw

    def process_small_turns(self, poc, poc_dyaw):
        new_poc = []
        new_poc_dyaw = []
        for idx, p in enumerate(poc):
            rounded = round2base(poc_dyaw[idx], np.pi/6)
            res = poc_dyaw[idx] - rounded
            if np.abs(res) > EPS:
                new_poc += [p, p]
                new_poc_dyaw += [rounded, res]
            else:
                new_poc += [p]
                new_poc_dyaw += [rounded]
        return new_poc, new_poc_dyaw

    def get_obstacles(self):
        obstacle_contours = {}
        for tag in Tags.DESCRIPTION_SET:
            bitmask = np.where(self.semantics_img == Tags.Hash[tag], 1, 0)
            padded_binary_mask = np.pad(bitmask, pad_width=1, mode='constant', constant_values=0)
            contours = measure.find_contours(padded_binary_mask, 0.5)
            contours = process_contours(contours)
            obstacle_contours.update({tag: contours})
        return obstacle_contours


class PathFinder(AStar):
    def __init__(self, img=None):
        if img == None:
            return
        self.img = img
        self.width = img.shape[1]
        self.height = img.shape[0]

    def update(self, img):
        self.img = img
        self.width = img.shape[1]
        self.height = img.shape[0]

    def heuristic_cost_estimate(self, current, goal):
        n1 = np.array(current)
        n2 = np.array(goal)
        v = n2 - n1
        norm = np.linalg.norm(v)
        # normed_v = v / norm
        yaw = np.arctan(n1[1]/n1[0])
        return norm

    def distance_between(self, n1, n2):
        n1 = np.array(n1)
        n2 = np.array(n2)
        v = n2 - n1
        norm = np.linalg.norm(v)
        delta_yaw = np.arctan(n2[1]/n2[0]) - np.arctan(n1[1]/n1[0])
        return 1

    def neighbors(self, node):
        neighbors = []
        x, y = node
        pts_adj = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        pts_diag = [(x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
        possible_pts = pts_adj + pts_diag
        for pt in possible_pts:
            if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height and self.img[pt[1], pt[0]] == 1:
                neighbors += [pt]
        return neighbors


def read_map_img(file_dir):
    data = np.load(file_dir, allow_pickle=True).tolist()
    return data['freemap'],data['semantics_map'], data['m_per_pixel']


def wrap2pi(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def round2base(num, base):
    return base*round(num/base)


def rad2deg(rad):
    deg = np.array(rad) * 180/np.pi
    return deg


def process_contours(contours, area_thresh=200):
    # draw contour in depth map?
    contour_results = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        bbox = get_bbox_from_contour(contour)
        area = np.abs(np.prod(np.abs(bbox[0, :]) - np.abs(bbox[2, :])))
        if area < area_thresh:
            continue
        contour = close_contour(contour)
        # obs = Obstable(contour, bbox)
        contour_results += [contour]
    return contour_results


def get_bbox_from_contour(flipped_contour):
    max_x = np.max(flipped_contour[:, 0])
    min_x = np.min(flipped_contour[:, 0])
    max_y = np.max(flipped_contour[:, 1])
    min_y = np.min(flipped_contour[:, 1])

    pt0 = [min_x, max_y]
    pt1 = [min_x, min_y]
    pt2 = [max_x, min_y]
    pt3 = [max_x, max_y]
    return np.array([pt0, pt1, pt2, pt3])


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def choose_endpoint(img, img_origin, min_range_pixel, max_range_pixel):
    if min_range_pixel > max_range_pixel:
        temp = max_range_pixel
        max_range_pixel = min_range_pixel
        min_range_pixel = temp
    # reverse filled box
    mask_filter = np.ones(img.shape)
    mask_filter[img_origin[1] - min_range_pixel: img_origin[1] + min_range_pixel,
                img_origin[0] - min_range_pixel: img_origin[0] + min_range_pixel] = 0
    masked_img = mask_filter * img
    # set max goal distance
    mask_filter = np.zeros(img.shape)
    mask_filter[img_origin[1] - max_range_pixel: img_origin[1] + max_range_pixel,
                img_origin[0] - max_range_pixel: img_origin[0] + max_range_pixel] = 1

    masked_img = mask_filter * masked_img

    free_coords = np.where((masked_img <= 1+EPS) & (1-EPS <= masked_img))
    if len(free_coords[0]) == 0:
        return None
    free_coords = np.concatenate((free_coords[1][:, None], free_coords[0][:, None]), axis=1)
    free_pt = random.choice(free_coords)
    x = free_pt[0]
    y = free_pt[1]
    # flip y-axis to make origin at bottom left
    # y = img.shape[0] - 1 - y
    return x, y


def signal_handler(signum, frame):
    raise Exception("Timed out!")


def set_timelimit(seconds):
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    return


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    files = os.listdir(data_dir)
    files.sort()
    for f in files:
        print(f)
        img, semantics_img,  m_per_pixel = read_map_img(data_dir/f)
        path_finder = PathFinder(img)
        start = np.where(img == 3)
        start = (start[0][0], start[1][0])

        # goals = []
        poi = []  # point of interest: first element is starting point, last element is goal
        paths = []
        instructions = []
        descriptions = []
        instruction_clusters = []
        description_clusters = []
        for i in range(args.instruction_num):
            n_failed_attempt = 0
            # print(i, "getting goal")
            while n_failed_attempt < MAX_FAIL_ATTEMPT:
                goal = choose_endpoint(img, start, int(args.min_goal_distance/m_per_pixel),
                                       int(args.max_goal_distance/m_per_pixel))
                # print('goal in meters: ', (np.array(goal) - start)*m_per_pixel)
                if (goal == None) or (img[goal[1], goal[0]] != 1):
                    print('Error: non-free point selected or no free pt')
                    n_failed_attempt += 1
                    continue
                # print(i, "getting path")
                print('GOAL:', goal)
                set_timelimit(5)
                try:
                    result = path_finder.astar(start, goal)
                except Exception:
                    n_failed_attempt += 1
                    print("Error: Astar timeout")
                    print("Choosing new goal")
                    continue
                    # break
                set_timelimit(0)
                if result != None:
                    path = list(path_finder.astar(start, goal))
                    path_pts = [list(x) for x in path]
                    path_pts = np.array(path_pts)
                else:
                    print("Error: None Path")
                    n_failed_attempt += 1
                    continue
                # print(i, "getting instruction")
                ins = InstructionGeneration((0, 0, 0), path_pts, semantics_img, m_per_pixel)
                instruction, instruction_cluster, poc = ins.get_instructions()
                description, description_cluster, des_poc = ins.get_description(poc)
                if len(instruction) > args.instruction_word_limit:
                    n_failed_attempt += 1
                    print("Error: Instruction too long")
                    continue
                break
            if n_failed_attempt >= MAX_FAIL_ATTEMPT:
                print("MAX_FAILED_ATTEMPT reached")
                # next file
                break

            all_goals = [path_pts[p] for p in poc]
            goals_in_meters = [(np.array(goal) - start)*m_per_pixel for goal in all_goals]
            yaw = ins.get_yaw()
            goal_yaws = [wrap2pi(yaw[p]) for p in poc]
            goal_pts = [tuple(goals_in_meters[i].tolist() + [goal_yaws[i]]) for i in range(len(all_goals))]
            poi += [goal_pts]
            paths += [path_pts]
            instructions += [instruction]
            descriptions += [description]
            instruction_clusters += [instruction_cluster]
            description_clusters += [description_cluster]
        if i != args.instruction_num - 1:
            continue
        print(f[:-4], "saving")
        data = {'data_id': f[:-4], 'poi': poi, 'instructions': instructions, 'instruction_clusters': instruction_clusters,
                'descriptions': descriptions, 'description_clusters': description_clusters,  'paths': paths}
        if not os.path.exists(output_dir):
            output_dir.mkdir()
        np.save(output_dir/(f[:-4] + '_inst.npy'), data)

    # print(path)
    # plt.imshow(img, origin='lower')
    # plt.plot(goal[0], goal[1], 'rx')
    # plt.plot(path_pts[:,0], path_pts[:,1], 'r')
    # fig, ax = plt.subplots()
    # ax.plot(ins.yaw)
    # ax.plot(ins.get_dyaw())

    # plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing input data")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--instruction_num", default=5, type=int, help="Instruction generated per freemap_img")
    parser.add_argument("--min_goal_distance", required=True, type=float,
                        help="Minimum range for goal selection (meters)")
    parser.add_argument("--max_goal_distance", required=True, type=float,
                        help="Maximum range for goal selection (meters)")
    parser.add_argument("--instruction_word_limit", default=40, type=int, help="World limit for instructions")
    args = parser.parse_args()
    main(args)