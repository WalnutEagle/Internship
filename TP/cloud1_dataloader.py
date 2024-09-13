import glob

import numpy as np

import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
#import carla
import cv2
import os



class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.img_list=[]
        self.data_list=[]
        # for dirs in ["./collection_data_final/"]:
        #     dir_name=os.listdir(dirs)
        #     for i in dir_name:
        #         self.data_dir = dirs+i
        #         #print(glob.glob(self.data_dir+'/*.jpg'))
        #         self.img_list =  self.img_list+glob.glob(self.data_dir+'/*.jpg')
        #         self.data_list = self.data_list+glob.glob(self.data_dir+'/*.npy')
        #self.data_list = np.load('d.npy', allow_pickle=True)#glob.glob(data_dir+'*.npy') #need to change to your data format
        self.data_dir="./data_13_11/"
        self.img_list =  self.img_list+glob.glob(self.data_dir+'/*.jpg')
        self.data_list = self.data_list+glob.glob(self.data_dir+'/*.npy')
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        self.transform2= transforms.Resize((96,96),antialias=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print(len(self.img_list))
        # Crop the image to the desired region
        #resized_image = chess[top_left_y:bottom_right_y, top_left_x:bottom_right_x]



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        data =  np.load(self.data_list[idx], allow_pickle=True)
        #print(self.img_list[idx])
        img = read_image(self.img_list[idx])
        img = img[:3,120:600,400:880] 
        # resized_image = img[:,:,80:560]
        normalized_image = self.normalize(img.float() / 255.0)
        #img = self.transform(data[0])#numpy is converted to tensor
        #image= self.transform2(img)#resize the image
        #print(type(resized_image))
        #converted_image=resized_image.permute(1,2,0)
        #final_data=self.front2bev(converted_image)
        #cv2.imwrite('transformed_image.jpeg',final_data)
        #print(type(converted_image))
        #image1=self.transform(converted_image)
        #image= self.transform2(image1)
        #cv2.imwrite('transformed_image.jpeg',image[0].numpy()*255)
        
        actions= torch.Tensor(data[:2])#convert actions to tensor
        locations=torch.Tensor(data[3:])
        location_final=torch.Tensor(np.array((locations[0]-locations[2],locations[1]-locations[3])))
        #print(affordance)
        return (normalized_image, actions,location_final)

    def front2bev(self,front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.
        
        input:
            front_view_image)96x96x3

        output:
            bev_image 96x96x3
        '''
        def homography_ipmnorm2g(top_view_region):#Top-view Perspective
                    src=np.float32([[0,0],[1,0],[0,1],[1,1]])
                    H_ipmnorm2g=cv2.getPerspectiveTransform(src,np.float32(top_view_region))
                    return H_ipmnorm2g

        bev=np.zeros((576,576,3))
        H,W=576,576
        top_view_region= np.array([[50,-25],[50,25],[0,25],[0,-25]])

        cam_xyz=[0.0,0.0,1.0]
        cam_yaw=0
        cam_pitch=0
        cam_roll=0

        width=576
        height=576
        fov=120

        focal=width/(2.0*np.tan(fov*np.pi/360.0))
        K=np.identity(3) #Projection Matrix
        K[0,0]=K[1,1]=focal
        K[0,2]=width/2.0
        K[1,2]=height/2.0
        #Translation and Rotation Matrix
        H_g2cam=np.array(carla.Transform(carla.Location(*cam_xyz),carla.Rotation(yaw=cam_yaw
                            ,pitch=cam_pitch,roll=cam_roll)).get_matrix())
        H_g2cam=np.concatenate([H_g2cam[:3,0:2],np.expand_dims(H_g2cam[:3,3],1)],1)
        #Multiplication of K and [R t]
        trans_mat=np.array([[0,1,0],[0,0,-1],[1,0,0]])
        temp_mat=np.matmul(trans_mat, H_g2cam)
        H_g2in=np.matmul(K,temp_mat)
        #Multiplication of K [R t] with top view region perspective
        H_ipmnorm2g=homography_ipmnorm2g(top_view_region)
        H_ipmnorm2in=np.matmul(H_g2in,H_ipmnorm2g)
        #Top-view projection
        S_in_inv=np.array([[1/np.float64(width),0,0],[0,1/np.float64(height),0],[0,0,1]])
        M_ipm2im_norm=np.matmul(S_in_inv,H_ipmnorm2in)

        #Visualisation
        M=torch.zeros(1,3,3)
        M[0]=torch.from_numpy(M_ipm2im_norm).type(torch.FloatTensor)

        linear_points_W=torch.linspace(0,1-1/W,W)
        linear_points_H=torch.linspace(0,1-1/H,H)

        base_grid=torch.zeros(H,W,3)
        base_grid[:,:,0]=torch.ger(torch.ones(H),linear_points_W)
        base_grid[:,:,1]=torch.ger(linear_points_H,torch.ones(W))
        base_grid[:,:,2]=1

        grid=torch.matmul(base_grid.view(H*W,3),M.transpose(1,2))
        lst=grid[:,:,2:].squeeze(0).squeeze(1).numpy()>=0
        grid=torch.div(grid[:,:,0:2],grid[:,:,2:])

        x_vals=grid[0,:,0].numpy()*width
        y_vals=grid[0,:,1].numpy()*height

        indicate_x1=x_vals<width
        indicate_x2=x_vals>0

        indicate_y1=y_vals<height
        indicate_y2=y_vals>0

        indicate=(indicate_x1*indicate_x2*indicate_y1*indicate_y2*lst)*1

        img=front_view_image

        for _i in range(H):
            for _j in range(W):
                _idx= _j+_i*W

                _x=int(x_vals[_idx])
                _y=int(y_vals[_idx])
                _indic=indicate[_idx]

                if _indic==0:
                    continue
                    # Ensure that _x and _y are within valid bounds
                
                bev[_i, _j] = img[_y, _x]
                #bev[_i,_j]=img[_y,_x]

        bev_image=np.uint8(bev)
        #cv2.imwrite('BEVimage.png',bev_image[508:570,252:252+72])
        #cv2.imshow('IMG1', np.uint8(bev_image))
        bev_new=bev_image[508:570,252:252+72]
        bgr_img = cv2.cvtColor(bev_new, cv2.COLOR_RGB2BGR)
        return bgr_img


def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True
            )