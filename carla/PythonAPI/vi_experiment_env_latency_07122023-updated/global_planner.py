import numpy as np

class Planner():
    def __init__(self, path_file):
        self.path_file = path_file
        data = np.flip(np.load(path_file, allow_pickle=True)[:],axis=0)
        # print(data)
        s = np.add.accumulate(np.linalg.norm(np.diff(data[:,:2], axis = 0), axis = 1))
        s = np.concatenate(([0], s))
        x = data[:,0:1]
        y = data[:,1:2]
        self.path_pts = np.concatenate((x, y, s[:,None]), axis=1) # x, y, s
        self.path_length = len(self.path_pts)
        self.path_length_value=0
        return
    

    def get_next_goal(self, pos, preview_s):
        pos = np.array(pos)
        closest_pt_id = self.get_closest_point(pos)
        # print('closest: ',closest_pt_id, pos)
        for i in range(closest_pt_id, self.path_length):
            ds = self.path_pts[i, 2] - self.path_pts[closest_pt_id, 2]
            if ds >= preview_s:
                # print('point goound: ', i, ds)
                return self.path_pts[i, [0,1]]
        # return final point if not found
        return self.path_pts[-1, 0], self.path_pts[-1, 1]
    

    def get_closest_point(self, pos):
        return np.argmin(np.linalg.norm(self.path_pts[:,0:2] - pos,  axis = 1))

    def get_path_length(self):
        for i in range(1,self.path_length):
            self.path_length_value+=np.linalg.norm(self.path_pts[i]-self.path_pts[i-1])
        return self.path_length_value

    def get_path_length_from_position(self, pos, dest):
        # print(self.path_pts.shape)
        # print(pos.shape)
        index = np.argwhere((self.path_pts[:,0:2]==pos).all(1)==True)[0][0]
        dest_index = -1
        print("Indices:",index, dest_index)
        # path_length_value = []
        # for i in range(index+1, self.path_length):
        #     path_length_value.append(np.linalg.norm(self.path_pts[i,0:2]-self.path_pts[i-1,0:2]))
        return self.path_pts[dest_index,2]-self.path_pts[index,2]
