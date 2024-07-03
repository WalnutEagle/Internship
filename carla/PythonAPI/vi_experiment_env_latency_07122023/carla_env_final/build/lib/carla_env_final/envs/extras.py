#action = obs[0][-1][:2]
        #reward += 1.0 - np.abs(action[0])
        # Apply control to the vehicle based on an actioneIt
        #print(action)




#print("Getting positive reward")
            # #print("Enter Actions")
            # #self.count_route_steps+=1
            # self.controller._control.speed = 1.5
            # self.controller._control.direction=self._run_global_policy()
            # self.world.player.apply_control(self.controller._control)
            # # if (self.minimap.player_pos.x in self.minimap.planner.path_pts[:,0:1]) and (self.minimap.player_pos.y in self.minimap.planner.path_pts[:,1:2]):
            # #     self.count_route_steps+=1
            
            # print("Inside global policy", self.controller._control)
            # #self.distance_covered=1.5*(time.time()-self.prev_time)
            # #self.controller.parse_events(self.client, self.world, self.clock)# Global Policy
            # #self.energy+=300*(time.time()-self.prev_time)
            # reward+=1.0
            








            # if (len(self.collision_hist)-1) != 0:
                #print("Collision History",self.col_count)
                # print("Collision History",self.collision_hist)
                # done = True
            
                #reward += -1500
        #else:
            #print("Getting no reward")
            # self.controller._control.speed = 0.0
            # self.world.player.apply_control(self.controller._control)# Local Policy-STOP
            # print("Inside local policy", self.controller._control)
            #obs = self._get_obs()
        
        # reward-=obs[1][2]/self.minimap.planner.get_path_length()
        # reward+=action[0][1]/2
        
        
        #print(type(reward))
        #info = dict()
        #self.distance_covered= 
        
        # print(action)
        # print(type(reward))
        #collision, pedestrian, distance_covered
        # 
        
        # reward+=100*(self.l2_distance)
        # print(type(reward))
        # print("reward from route completion",(reward))
        # if obs.get("collision_detected") != 0:
        
        #print(self.prev_x,self.prev_y)
        # self.l2_distance=obs.get("distance_covered")