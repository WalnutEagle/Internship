import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from freemap_extraction import Tags

DATA_DIR = Path("/home/tzm/Projects/2_Lab/data/carla/vi_dataset_4/Town10HD_0/")
SAVE_DIR = DATA_DIR/"freemap_test/"
EPS = 10e-4


class Freemap():
    def __init__(self, points=None, tags=None):
        self.points = points
        self.tags = tags
        return

    def load(self, file_dir):
        data = np.load(file_dir, allow_pickle=True).tolist()
        self.__init__(data['camera_pose_info'], data['points'], data['tags'])
        return

    def save(self, file_dir):
        data = {'camera_pose_info': self.camera_post, 'points': self.points.astype(
            np.single), 'tags': self.tags.astype(np.uint8)}
        np.save(str(SAVE_DIR/file_dir), data)
        return

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(0, 0, 'rx')
        for tag in Tags.Hash:
            pts = fm.points[self.tags == Tags.Hash[tag]]
            ax.plot(pts[:, 0], pts[:, 1], '.', color=tuple(np.array(Tags.Colors[Tags.Hash[tag]])/255))

    def get_img(self, m_per_pixel, max_distance, padding=5):
        n_pixel = int(max_distance/m_per_pixel)*2+2
        img = np.zeros((n_pixel, n_pixel))
        semantics_img = np.zeros((n_pixel, n_pixel))
        rounded_points = (np.round((self.points[:, 0:2] + max_distance)/m_per_pixel)).astype(int)
        free_pts_idx = []
        free_pts_tags = []
        nonfree_pts_idx = []
        nonfree_pts_tags = []
        tag_hash_inv = list(Tags.Hash.keys())
        for idx, tag in enumerate(self.tags):
            tag_name = tag_hash_inv[tag]
            if tag_name in Tags.FREE_SET:
                free_pts_idx += [idx]
                free_pts_tags += [tag]
            # else:
            elif tag_name in Tags.NONFREE_SET:
                nonfree_pts_idx += [idx]
                nonfree_pts_tags += [tag]

        free_point = rounded_points[free_pts_idx]
        nonfree_point = rounded_points[nonfree_pts_idx]
        # print(len(free_pts_idx))
        # print(len(nonfree_pts_idx))
        # print(img.shape)

        # patch for camera blind spot
        # for x in range(530, 970):
        #     for y in range(530, 970):
        #         if img[y][x] == 0:
        #             img[y][x] = 1

        for x in range(550, 950):
            for y in range(550, 950):
                if img[y][x] == 0:
                    img[y][x] = 1

        # fill in rest:
        for idx, pt in enumerate(free_point):
            img[pt[1]-padding:pt[1]+padding, pt[0]-padding: pt[0]+padding] = 1
            semantics_img[pt[1]-padding:pt[1]+padding, pt[0]-padding: pt[0]+padding] = free_pts_tags[idx]

        for idx, pt in enumerate(nonfree_point):
            img[pt[1]-padding:pt[1]+padding, pt[0]-padding:pt[0]+padding] = 2
            semantics_img[pt[1]-padding:pt[1]+padding, pt[0]-padding: pt[0]+padding] = nonfree_pts_tags[idx]

        # add Ego
        ego_img_coord = np.round((np.zeros((2)) + max_distance)/m_per_pixel).astype(int)
        # print(ego_img_coord)
        img[ego_img_coord[1]][ego_img_coord[0]] = 3
        print('end of get_img')
        # fig2, ax2  = plt.subplots()
        # ax2.imshow(img, origin='lower') # origin at left-bottom corner
        #  ax.plot(fm.points[:,0], fm.points[:,1], color=tuple(np.array([Tags.Colors[tag] for tag in fm.tags])/255))
        # plt.show()
        return img, semantics_img


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    files = os.listdir(data_dir)
    for f in files:
        print(f)
        f_dir = data_dir/f
        fm = Freemap()
        fm.load(file_dir=f_dir)

        # fig, ax  = plt.subplots()
        # ax.plot(0,0,'rx')
        # for tag in Tags.Hash:
        #     pts = fm.points[fm.tags == Tags.Hash[tag]]
        #     ax.plot(pts[:,0], pts[:,1], '.',color = tuple(np.array(Tags.Colors[Tags.Hash[tag]])/255))
        # plt.show()
        img, semantics_img = fm.get_img(args.m_per_pixel, args.max_distance, args.padding)
        # fig2, ax2  = plt.subplots()
        # ax2.imshow(img, origin='lower') # origin at left-bottom corner
        # ax.plot(fm.points[:,0], fm.points[:,1], color=tuple(np.array([Tags.Colors[tag] for tag in fm.tags])/255))
        data = {'freemap': img, 'semantics_map': semantics_img,
                'm_per_pixel': args.m_per_pixel, 'padding': args.padding}
        # SAVE IMAGE
        # plt.show()

        if not os.path.exists(output_dir):
            output_dir.mkdir()
        np.save(output_dir/f, data)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, required=True, help="Directory of freemap .npy files")
    parser.add_argument("--output_dir", default=None, required=True, help="Output directory")
    parser.add_argument("--padding", default=5, help="Number of pixel padding per dot of object")
    parser.add_argument("--m_per_pixel", default=0.02, help="Meter per image pixel")
    parser.add_argument("--max_distance", default=15, help="Maximum view distance in meters")
    args = parser.parse_args()
    main(args)