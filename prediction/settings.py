

# from predictor import VisualizationDemo
# from demo_w_gRPC import config_camera, setup_cfg


systemID = "BHS" # or  SBD
prediction_model = "efficientdet" # efficientdet or maskrcnn

check_button = False
force_in_button = False
conveyable = True
cameraId = 1 # camera_1 :920312073100
wait_for_user = False

# command
stopAnalyse = False

# point cloud
width, height = 640, 480
portrait = True



# discarded
classifications = ['suitcase',          #1
                   'soft_bag',          #2
                   'wheel',             #3
                   'extended_handle',   #4
                   'person',            #5
                   'tray',              #6
                   'upright_suitcase',  #7
                   'spilled_bag',       #8
                   'sphere_bag',        #9
                   'documents',         #10
                   'bag_tag',           #11
                   'strap_around_bag',  #12
                   'stroller',          #13
                   'golf_bag',          #14
                   'surf_equipment',    #15
                   'sport_equipment',   #16
                   'music_equipment',   #17
                   'plastic_bag',       #18
                   'shopping_bag',      #19
                   'wrapped_bag',       #20
                   'umbrella',          #21
                   'storage_container', #22
                   'box',               #23
                   'big_wheel',         #24
                   'laptop_bag',        #25
                   'tube',              #26
                   'pet_container',     #27
                   'ski_equipment',     #28
                   'tripod',            #29
                   'child_safety_car_seat', #30
                   'tool_box',          #31
                   'very_small_parcel', #32
                   'bingo_sticker']     #33
                    # upright_suitcase missing

event_list = ['wheel at front', 
            'tray is required', 
            'upright position', 
            'wrong orientation', 
            'out of conveyor', 
            'extended handle',
            'bag is open',
            'document is detected',
            'unauthorized object',
            'multi bag',
        ]
