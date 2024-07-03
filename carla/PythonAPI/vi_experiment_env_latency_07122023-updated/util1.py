import glob
import os
import sys
import numpy as np
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import queue
except ImportError:
    import Queue as queue


import carla

from libcarla.command import SpawnActor as SpawnActor
from libcarla.command import SetAutopilot as SetAutopilot
from libcarla.command import SetVehicleLightState as SetVehicleLightState
from libcarla.command import FutureActor as FutureActor

WEATHER = {'ClearNoon':carla.WeatherParameters.ClearNoon,
    'ClearSunset': carla.WeatherParameters.ClearSunset,
    'WetNoon': carla.WeatherParameters.WetNoon,
    'HardRainNoon': carla.WeatherParameters.HardRainNoon,
    'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
    'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
    'HardRainSunset': carla.WeatherParameters.HardRainSunset}


def get_image(image, _type=None):
    if _type == 'semantic_segmentation':
        image.convert(carla.ColorConverter.CityScapesPalette)

    _array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    _array = np.reshape(_array, (image.height, image.width, 4))
    _array = _array[:, :, :3]
    _array = _array[:, :, ::-1]

    # if _type == 'depth':
    #     depth = np.float32(_array)
    #     normalized = (depth[:,:,0] + depth[:,:,1]*256 + depth[:,:,2]*256*256) / (256 * 256 * 256 - 1)
    #     in_meters = 1000 * normalized
    #     return in_meters
    # else:
    #     return _array
    
    return _array



def get_traffic_manager(client, port=4500):
    tm = client.get_trafficmanager(port)
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.set_hybrid_physics_mode(True)
    tm.set_synchronous_mode(True)
    return tm


# -------------------Jim

def spawn_vehicles(client, world, number_of_vehicles, synchronous_master, traffic_manager, car_lights_on=False):
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    vehicles_list = []
    spawn_points = world.get_map().get_spawn_points()
    batch = []
    vehicles_bp = world.get_blueprint_library().filter('vehicle.*')
    vehicles_bp_4 = [x for x in vehicles_bp if int(x.get_attribute('number_of_wheels')) == 4] # only use 4 wheel vehicles
    vehicles_bp_2 = [x for x in vehicles_bp if int(x.get_attribute('number_of_wheels')) == 2] # only use 2 wheel vehicles
    # hero = args.hero
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        if random.random() > 0.1:
            blueprint = random.choice(vehicles_bp_4)
        else:
            blueprint = random.choice(vehicles_bp_2)

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            print(response.error)
        else:
            vehicles_list.append(response.actor_id)

    # Set automatic vehicle lights update if specified
    if car_lights_on:
        all_vehicle_actors = world.get_actors(vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)

    return vehicles_list





# -----------------
# Spawn vehicles
# -----------------


# def spawn_vehicles(client, world, N, traffic_manager, vehicles_list):
#     spawn_pts = world.get_map().get_spawn_points()
#     vehicles_bp = world.get_blueprint_library().filter('vehicle.*')
#     if len(spawn_pts) < N:
#         print("vehicle spawn_pts < N")
#         return False

#     batch = []
#     for idx in range(N):
#         bp = random.choice(vehicles_bp)
#         if bp.has_attribute('color'):
#             color = random.choice(bp.get_attribute('color').recommended_values)
#             bp.set_attribute('color', color)
#         if bp.has_attribute('driver_id'):
#             driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
#             bp.set_attribute('driver_id', driver_id)
#         bp.set_attribute('role_name', 'autopilot')

#         batch.append(SpawnActor(bp, spawn_pts[idx])
#                      .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

#     for response in client.apply_batch_sync(batch, True):  # True is for synchronize mode?
#         if response.error:
#             print(response.error)
#         else:
#             vehicles_list.append(response.actor_id)
#     return True

# -----------------
# Spawn walkers
# -----------------

regularped_ids = ['0001','0002','0003','0004','0005','0006','0007','0008','0009','0010',
                    '0011','0012','0013','0014','0015','0016','0017','0018','0019','0020','0021']

visuallyimpairedped_train_ids = ['0022','0023','0024','0025','0026','0027','0028','0029','0030',
                                    '0031','0032','0033','0034','0035','0036']

visuallyimpairedped_test_ids = ['0325','0326','0327','0328','0329','0330','0331','0332','0333',
                                    '0334','0335','0336','0337','0338','0339']

visuallyimpairedped_all_ids = visuallyimpairedped_train_ids + visuallyimpairedped_test_ids

wheelchairped_train_ids = ['0037','0038','0039','0040','0041','0042','0043','0044','0045','0046',
                            '0047','0048','0049','0050','0051','0052','0053','0054','0055','0056',
                            '0057','0058','0059','0060','0085','0086','0087','0088','0089','0090',
                            '0091','0092','0093','0094','0095','0096','0109','0110','0111','0112',
                            '0113','0114','0115','0116','0117','0118','0119','0120','0121','0122',
                            '0123','0124','0125','0126','0127','0128','0129','0130','0131','0132',
                            '0157','0158','0159','0160','0161','0162','0163','0164','0165','0166',
                            '0167','0168']
# ncos: new color old shape
wheelchairped_ncos_ids = ['0061','0062','0063','0064','0065','0066','0067','0068','0069','0070','0071',
                            '0072','0073','0074','0075','0076','0077','0078','0079','0080','0081','0082',
                            '0083','0084','0097','0098','0099','0100','0101','0102','0103','0104','0105',
                            '0106','0107','0108','0133','0134','0135','0136','0137','0138','0139','0140',
                            '0141','0142','0143','0144','0145','0146','0147','0148','0149','0150','0151',
                            '0152','0153','0154','0155','0156','0169','0170','0171','0172','0173','0174',
                            '0175','0176','0177','0178','0179','0180']

# ncns: new color new shape
wheelchairped_ncns_ids = ['0205','0206','0207','0208','0209','0210','0211','0212','0213','0214','0215',
                            '0216','0217','0218','0219','0220','0221','0222','0223','0224','0225','0226',
                            '0227','0228','0241','0242','0243','0244','0245','0246','0247','0248','0249',
                            '0250','0251','0252','0277','0278','0279','0280','0281','0282','0283','0284',
                            '0285','0286','0287','0288','0289','0290','0291','0292','0293','0294','0295',
                            '0296','0297','0298','0299','0300','0313','0314','0315','0316','0317','0318',
                            '0319','0320','0321','0322','0323','0324']

# ocns: old color new shape
wheelchairped_ocns_ids = ['0181','0182','0183','0184','0185','0186','0187','0188','0189','0190','0191',
                            '0192','0193','0194','0195','0196','0197','0198','0199','0200','0201','0202',
                            '0203','0204','0229','0230','0231','0232','0233','0234','0235','0236','0237',
                            '0238','0239','0240','0253','0254','0255','0256','0257','0258','0259','0260',
                            '0261','0262','0263','0264','0265','0266','0267','0268','0269','0270','0271',
                            '0272','0273','0274','0275','0276','0301','0302','0303','0304','0305','0306',
                            '0307','0308','0309','0310','0311','0312']

wheelchairped_all_ids = wheelchairped_train_ids+wheelchairped_ncos_ids+wheelchairped_ncns_ids+wheelchairped_ocns_ids


def spawn_walkers(client, world, N, percent_disabled, percent_walking, percent_crossing, all_id_dict):
    #np.random.seed(42)
    spawn_pts = []


    for i in range(N):
        spawn_pt = carla.Transform()
        random_loc = world.get_random_location_from_navigation()
        if random_loc != None:
            spawn_pt.location = random_loc
            spawn_pt.location.z=0.5999999642372131
            spawn_pts.append(spawn_pt)
    print('walker spawn pts: ', len(spawn_pts))

    batch = []
    walkers_list = []
    walker_speed = []
    for pt in spawn_pts:
        # walker_bp = random.choice(world.get_blueprint_library().filter('walker'))

        # if (random.random() < 0.1):
        #     if (random.random() < 0.2):
        #         disabled_id = random.choice(visuallyimpairedped_all_ids)
        #     else:
        #         disabled_id = random.choice(wheelchairped_all_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+disabled_id)[0]
        #     if_disabled = True
        #     #print("disable id ", disabled_id)
        # else:
        #     regular_id = random.choice(regularped_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]
        #     #print("regular id ", regular_id)

        regular_id = random.choice(regularped_ids)
        walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]


        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if (random.random() < percent_walking):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, pt))
    
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(response.error)
        else:
            walkers_list.append(response.actor_id)

    batch = []
    controller_list = []
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for walker in walkers_list:
        batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(response.error)
        else:
            controller_list.append(response.actor_id)

    all_id_dict.update({'walkers': walkers_list})
    all_id_dict.update({'controllers': controller_list})

    world.tick()

    world.set_pedestrians_cross_factor(percent_crossing)
    for id in controller_list:
        controller = world.get_actor(id)
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        # all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    # for walker in walkers_list:
    #     print(world.get_actor(walker).get_velocity())
    return True



def sample_around(loc, radius = 50, N = 100):
    z = 0.5999999642372131 # fixed spawn height
    #print("Dense spawn z")
    #np.random.seed(42)
    np.random.seed(42)
    xs = np.random.uniform(loc[0]-radius, loc[0]+radius, N)
    ys = np.random.uniform(loc[1]-radius, loc[1]+radius, N)
    coord = np.stack((xs, ys, [z]*N), axis = 1)
    return coord


def spawn_walkers_dense(client, world, N, loc_center, percent_disabled, percent_walking, percent_crossing, all_id_dict,count,s):
    spawn_pts = []


    sampled_points = sample_around(loc_center, radius = 15, N=N)

    for i in range(N):
        pt = sampled_points[i, :]
        spawn_point = carla.Transform()
        print("z dense spawn point",pt[2])
        print(pt)
        loc = carla.Location(x = pt[0], y = pt[1], z = pt[2])
        spawn_point.location = loc
        spawn_pts.append(spawn_point)


    # for i in range(N):
    #     spawn_pt = carla.Transform()
    #     random_loc = world.get_random_location_from_navigation()
    #     if random_loc != None:
    #         spawn_pt.location = random_loc
    #         spawn_pts.append(spawn_pt)
    print('walker spawn pts: ', len(spawn_pts))

    batch = []
    walkers_list = []
    walker_speed = []

    # Fixed spawn point for pedestrian coming from opposite direction as ego vehicle
    # ped_spawn_pt = carla.Transform()
    # ped_spawn_pt.location = carla.Location(x=-7.51,y=59.86, z=0.5999999642372131)
    # spawn_pts = [ped_spawn_pt]
    for pt in spawn_pts:
        # walker_bp = random.choice(world.get_blueprint_library().filter('walker'))

        # if (random.random() < 0.1):
        #     if (random.random() < 0.2):
        #         disabled_id = random.choice(visuallyimpairedped_all_ids)
        #     else:
        #         disabled_id = random.choice(wheelchairped_all_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+disabled_id)[0]
        #     if_disabled = True
        #     #print("disable id ", disabled_id)
        # else:
        #     regular_id = random.choice(regularped_ids)
        #     walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]
        #     #print("regular id ", regular_id)

        regular_id = random.choice(regularped_ids)
        walker_bp = world.get_blueprint_library().filter('walker.pedestrian.'+regular_id)[0]

        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if (random.random() < percent_walking):
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, pt))
    
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(response.error)
        else:
            walkers_list.append(response.actor_id)

    batch = []
    controller_list = []
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for walker in walkers_list:
        batch.append(SpawnActor(controller_bp, carla.Transform(), walker))
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(response.error)
        else:
            controller_list.append(response.actor_id)

    all_id_dict.update({'walkers': walkers_list})
    all_id_dict.update({'controllers': controller_list})

    world.tick()
    world.set_pedestrians_seed(42) 
    world.set_pedestrians_cross_factor(percent_crossing)
    l=0
    for id in controller_list:
        controller = world.get_actor(id)
        controller.start()
        if count==0:
            m=world.get_random_location_from_navigation()
            #m = carla.Location(x=-29,y=56.5,z=0)
            m.z=0.5999999642372131
            s.append(m)
            print(s[-1].x,s[-1].y)
            controller.go_to_location(s[-1])
        else:
            print(l)
            print(s[0],'Hi')
            print(s[l].x,s[l].y)
            controller.go_to_location(s[l])
            l+=1
        # all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    # for walker in walkers_list:
    #     print(world.get_actor(walker).get_velocity())
    return True,s
