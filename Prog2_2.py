# Move a robot along a line given a start and end point by steps
# This macro shows different ways of programming a robot using a Python script and the RoboDK API

#sys.path.append("C:\Users\Sanket\AppData\Local\Programs\Python\Python36")
#---------------------------------------------------
#--------------- PROGRAM START ---------------------
from robolink import *    # API to communicate with RoboDK for simulation and offline/online programming
from robodk import *      # Robotics toolbox for industrial robots


from handtracking_master import *
from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

#sys.path.append('..')
detection_graph, sess = detector_utils.load_inference_graph()


# Initialize the RoboDK API
RDK = Robolink()

# turn off auto rendering (faster)
RDK.Render(False)

# Promt the user to select a robot (if only one robot is available it will select that robot automatically)
robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)

# Turn rendering ON before starting the simulation
RDK.Render(True)

# Retrieve the robot reference frame
reference = robot.Parent()

# Use the robot base frame as the active reference
robot.setPoseFrame(reference)

# Default parameters:
P_START = [475, 0, 320]    # Start point with respect to the robot base frame
P_END   = [575, 0, 320]    # End point with respect to the robot base frame

NUM_POINTS  = 2                # Number of points to interpolate

"""
# Function definition to create a list of points (line)
def MakePoints(xStart, xEnd, numPoints):
    #Generates a list of points
    if len(xStart) != 3 or len(xEnd) != 3:
        raise Exception("Start and end point must be 3-dimensional vectors")
    if numPoints < 2:
        raise Exception("At least two points are required")
    
    # Starting Points
    pt_list = []
    x = xStart[0]
    y = xStart[1]
    z = xStart[2]

    # How much we add/subtract between each interpolated point
    x_steps = (xEnd[0] - xStart[0])/(numPoints-1)
    y_steps = (xEnd[1] - xStart[1])/(numPoints-1)
    z_steps = (xEnd[2] - xStart[2])/(numPoints-1)

    # Incrementally add to each point until the end point is reached
    for i in range(numPoints):
        point_i = [x,y,z] # create a point
        #append the point to the list
        pt_list.append(point_i)
        x = x + x_steps
        y = y + y_steps
        z = z + z_steps
    return pt_list
"""


#if __name__ == '__main__':

parser = argparse.ArgumentParser()
parser.add_argument(
'-sth',
'--scorethreshold',
dest='score_thresh',
type=float,
default=0.2,
help='Score threshold for displaying bounding boxes')
parser.add_argument(
'-fps',
'--fps',
dest='fps',
type=int,
default=1,
help='Show FPS on detection/display visualization')
parser.add_argument(
'-src',
'--source',
dest='video_source',
default=0,
help='Device index of the camera.')
parser.add_argument(
'-wd',
'--width',
dest='width',
type=int,
default=320,
help='Width of the frames in the video stream.')
parser.add_argument(
'-ht',
'--height',
dest='height',
type=int,
default=180,
help='Height of the frames in the video stream.')
parser.add_argument(
'-ds',
'--display',
dest='display',
type=int,
default=1,
help='Display the detected images using OpenCV. This reduces FPS')
parser.add_argument(
'-num-w',
'--num-workers',
dest='num_workers',
type=int,
default=4,
help='Number of workers.')
parser.add_argument(
'-q-size',
'--queue-size',
dest='queue_size',
type=int,
default=5,
help='Size of the queue.')
args = parser.parse_args()


cap = cv2.VideoCapture(args.video_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

start_time = datetime.datetime.now()
num_frames = 0
im_width, im_height = (cap.get(3), cap.get(4))
# max number of hands we want to detect/track
num_hands_detect = 2

cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

while True:
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    ret, image_np = cap.read()
    # image_np = cv2.flip(image_np, 1)
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

    # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
    # while scores contains the confidence for each of these boxes.
    # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

    boxes, scores = detector_utils.detect_objects(image_np,
                                                  detection_graph, sess)
    score_thresh = 0.27
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            mid_x = int((boxes[i][3] - boxes[i][1])* im_width)
            mid_y = int((boxes[i][2] - boxes[i][0])* im_height)
            #print(str(mid_x) + ", " + str(mid_y))
    # draw bounding boxes on frame
    detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                     scores, boxes, im_width, im_height,
                                     image_np)

    # Calculate Frames per second (FPS)
    num_frames += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time

    if (args.display > 0):
        # Display FPS on frame
        if (args.fps > 0):
            detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                             image_np)

        cv2.imshow('Single-Threaded Detection',
                   cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        print("frames processed: ", num_frames, "elapsed time: ",
              elapsed_time, "fps: ", str(int(fps)))
    #------------------------------------------------------------------
    P_START = P_END
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            mid_x = int((boxes[i][3] - boxes[i][1])* im_width)
            mid_y = int((boxes[i][2] - boxes[i][0])* im_height)        
            #print(str(x) + "," + str(y))
            #P_END = [320, (mid_x+285)*1.5, (160-mid_y)*1.5]
            P_END = [320, (80-mid_y)*1.5, (mid_x+80)*1.5]
            print(P_END)
            print("midx and midy pos: " + str(mid_x) + "," + str(mid_y))


        
    #-------------------------------------------------------------------
    #calculate points

    if len(P_START) != 3 or len(P_END) != 3:
        raise Exception("Start and end point must be 3-dimensional vectors")
    if NUM_POINTS < 2:
        raise Exception("At least two points are required")

    
    # Starting Points
    pt_list = []
    x = P_START[0]
    y = P_START[1]
    z = P_START[2]

    # How much we add/subtract between each interpolated point
    x_steps = (P_END[0] - P_START[0])/(NUM_POINTS-1)
    y_steps = (P_END[1] - P_START[1])/(NUM_POINTS-1)
    z_steps = (P_END[2] - P_START[2])/(NUM_POINTS-1)

    # Incrementally add to each point until the end point is reached
    for i in range(NUM_POINTS):
        point_i = [x,y,z] # create a point
        #append the point to the list
        pt_list.append(point_i)
        x = x + x_steps
        y = y + y_steps
        z = z + z_steps
    #---------------------------------------------------------------------

    
    POINTS = pt_list

    # Automatically delete previously generated items (Auto tag)
    list_items = RDK.ItemList() # list all names
    for item in list_items:
        if item.Name().startswith('Auto'):
            item.Delete()

    
    # Abort if the user hits Cancel
    if not robot.Valid():
        quit()


    # get the current orientation of the robot (with respect to the active reference frame and tool frame)
    pose_ref = robot.Pose()
    print("before moving orientation: ")
    print(Pose_2_TxyzRxyz(pose_ref))
    #print("-----------------------")
    # a pose can also be defined as xyzwpr / xyzABC
    #pose_ref = TxyzRxyz_2_Pose([100,200,300,0,0,pi])

    
    #-------------------------------------------------------------
    # Option 3: Move the robot using the Python script and detect if movements can be linear
    # This is an improved version of option 1
    #
    # We can automatically force the "Create robot program" action using a RUNMODE state
    # RDK.setRunMode(RUNMODE_MAKE_ROBOTPROG)

    # Iterate through all the points
    ROBOT_JOINTS = None
    for i in range(NUM_POINTS):
        # update the reference target with the desired XYZ coordinates
        pose_i = pose_ref
        pose_i.setPos(POINTS[i])
        
        # Move the robot to that target:
        if i == 0:
            # important: make the first movement a joint move!
            robot.MoveJ(pose_i)
            ROBOT_JOINTS = robot.Joints()
        else:
            # test if we can do a linear movement from the current position to the next point
            if robot.MoveL_Test(ROBOT_JOINTS, pose_i) == 0:
                robot.MoveL(pose_i)
            else:
                robot.MoveJ(pose_i)
                
            ROBOT_JOINTS = robot.Joints()
            
    pose_ref2 = robot.Pose()
    print("after moving orientation: ")
    print(Pose_2_TxyzRxyz(pose_ref2))
    print("-------------------------")
    


