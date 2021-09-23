import os
import pandas as pd
import numpy as np
from maze import Maze

# global dictionaries for robot movement and sensing
max_moves=3
directions = ['u','r','d','l']

rotations = {'u': {'l':-90, 'u':0, 'r':90, 'd':180},
             'l': {'l':0, 'u':90, 'r':180, 'd':-90},
             'r': {'l':180, 'u':-90, 'r':0, 'd':90},
             'd': {'l': 90, 'u': 180, 'r': -90, 'd': 0}}

next_heading = {'u': {-90:'l', 90:'r'},
                'l': {-90:'d', 90:'u'},
                'r': {-90:'u', 90:'d'},
                'd': {-90:'r', 90:'l'}}

dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u']}

dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0]}

wall_mask = {'u': 1, 'r': 2, 'd': 4, 'l': 8,}

wall_mask_rev= {'u': 4, 'r': 8, 'd': 1, 'l': 2}

moving_modes = ['explore_to_target','explore_from_target','go']

exploring_paths = {0:[],
                   1:[['br','tl','tr']],
                   2:[['bl']],
                   3:[['br','tl','tr'],['bl']],
                   4:[['tl'],['cc'],['tr'],['cc'],['br'],['cc']],
                   5:[['bl'],['cc'],['bl'],['cc']],
                   6:[['bl'],['cc'],['bl'],['cc'],['bl'],['cc']],
                   7: [['tl'], ['tr'], ['br'], ['bl']]}

class Robot(object):
    """
    The class maintains robot position and orientation in the maze and provides methods
    for the robot to explore and move into the maze.

    Attributes
    ----------
    maze_dim: integer.
        Size of the maze, received as input parameter. Referred as 'N' in following notes
    location: list of two integers.
        Position of the robot in the maze. Bottom left is [0,0], bottom right is [N,0]
    heading: str.
        Single character string for current robot orientation ('u'=up, 'd'=down, 'l'=left, 'r'=right)
    target: list of 2-sized tuple of integers.
        Each tuple represents a box in the maze. List of boxes that are the target for maze navigation (maze center)
    walls: N x N numpy array of integers.
        Store the robot-internal representation of the maze, based on history of info collected from sensors
        Walls[i,j] provides a 4 bit based representation of the walls in maze box [i,j]. 1 bit mean missing wall.
        0, 15: 4 walls and no walls respectively
        1,2,4,8: box exit only on top, right, bottom and left respectively
        Note that walls include assumptions made by the robot, so the representation changes at each sensing
    known_vert: numpy array of boleean as integers (0,1).
        Store the information about the verified knowledge about a specific vertical wall.
        known_vert[i,j]=1 means that the robot sensors have verified presence or absence of the wall on the left of box i,j
    known_hor: as known_vert for horizontal walls
    mode: string in ['explore_to_target','explore_from_target','go'].
        The mode the robot is using to select moves, according to following definitions:
        'explore_to_target': robot is in run 0 and is making first exploration to reach center boxes
        'explore_from_target': robot is in run 0, has reached the center but is further exploring the maze before
                                moving to run 1. Exploration is driven by exp_path attribute
        'go': robot is in run 1 and is trying to reach the center
    exp_path: integer.
        Defines the procedure the robot is using to move in 'explore_from_target' mode.
        0: center -> 'go' (Exploration finishes immediately when center of maze is reached the first time)
        1: center -> any corner 'go' (From center, the exploration continues toward any corner, then finishes)
        2: center -> 0,0 -> 'go'  (From center, to bottom left, then 'go' mode)
        3: center -> any corner -> 0,0 -> 'go'
        4: center -> 0,n -> center -> n,n -> center -> n,0 -> center -> 'go'
        5: center -> 0,0 -> center -> 0,0 -> center -> 'go'
        6: center -> 0,0 -> center -> 0,0 -> center -> 0,0 -> center -> 'go'
    exp_target: list of tuples.
        The next target the robot must reach to proceed exploration
    exp_step = integer.
        Step ongoing in the selected exploration modes
    flood: N x N numpy array of integers.
        Flood [i,j] represents the distance (in number of boxes) from location to current target, based on walls values.
    restart: Boolean.
        Used to define if at next move flood will have to be recalculated from scratch.
        True: Need to recalculate the whole flood matrix, normally becouse target for navigation has changed
        False: We can recalculate only the part affected by info just received from sensors

    Methods
    -------
    next_move(self, sensors)
        Return to class user next robot move, based on sensors input and on internal robot status
    init_known_walls(self)
        initialize known_vert and known_hor attributes
    init_walls(self):
        initialize walls attributes


    Internal methods
    -------
    __calculate_next_move(self, sensors)
        Called by next move for actual move calculation, including best path algorithm (flooding) logic
    __explore(self,  direction, mx_step)
        Calculate best moving option from present location in a specific direction, with limit in max move length
    __move_robot(self,rotation, movement)
        Perform actual robot move, updating location and heading
    __update_walls(self,sensing):
        updates walls, known_vert and known_hor based on sensor input
    """

    def __init__(self, maze_dim):
        """
        Initialize class attributes
        """

        self.location = [0, 0]
        self.heading = 'u'
        self.maze_dim = maze_dim
        hd=int(maze_dim/2)
        self.target = [(hd-1,hd-1),(hd-1,hd),(hd,hd-1),(hd,hd)]
        self.exp_path=5
        self.exp_target,self.exp_step = self.init_exp_target()
        self.init_walls() #initializes walls attribute
        self.init_known_walls() #initializes known_vert and known_hor attributes
        self.mode = "explore_to_target"
        self.flood = flooding(maze_dim, self.walls, self.target)
        self.restart = False


    def next_move(self, sensors):
        """
        Determines the next move the robot should make,based on the input from the sensors
        and on internal robot status.

        Parameter:
        -------
            Sensor: list of 3 integers, represnting distance from next wall perceived from left, front and right sensors

        Return:
        -------
        rotation:
            robot rotation (if any), as a number: 0 for no rotation, +90 for a 90-degree rotation clockwise,
            and -90 for a 90-degree rotation counterclockwise.
        movement:
            number of squares the robot is supposed to move. Movement is backwards if the number is negative.

        Special return cases:
        -------
        'Reset','Reset':
            is returned to complete run 0 and start run 1. The robot is assumed to start from bottom left maze corner
        0,0:
            is returned to move to next target while in exploration mode
        """

        #print('From:', self.mode, self.location)
        reset=False
        #'explore_to_target' mode: Run 0, the robot is trying to reach the center for the first time
        if self.mode== 'explore_to_target':
            rotation, movement= self.__calculate_next_move(sensors, self.target, reset, mx_step=max_moves)
            if (rotation,movement)==(0,0):
                reset = True
                if self.exp_target==[]:
                    self.mode = 'go'
                    self.heading = 'u'
                    self.location=[0,0]
                    return 'Reset','Reset'
                else:
                    self.exp_step=0
                    self.mode = 'explore_from_target'

        #'explore_from_target' mode: Run 0, the robot has reached the center at least once but is still exploring as per 'exp_path'
        if self.mode== 'explore_from_target':
            rotation, movement = self.__calculate_next_move(sensors, self.exp_target[self.exp_step], reset, mx_step=max_moves)
            if (rotation,movement)==(0,0):
                reset=True
                self.exp_step += 1
                if(len(self.exp_target)<=self.exp_step):
                    self.mode = 'go'
                    self.heading = 'u'
                    self.location=[0,0]
                    return 'Reset','Reset'
                else:
                    rotation, movement = self.__calculate_next_move(sensors, self.exp_target[self.exp_step], reset,mx_step=max_moves)

        #'go' mode: Run 1, the robot is trying to reach the center on the quickest path
        if self.mode== 'go':
            if self.location==[0,0] and self.heading=='u':
                reset=True
            rotation, movement= self.__calculate_next_move(sensors, self.target, reset, mx_step=max_moves)
        #print('To:', self.location)
        return rotation, movement


    def __calculate_next_move(self, sensors, target, reset, mx_step):
        """
        The method perform actual next move calculation, after updating internal maze picture based on sensors info
        Next move calculation is based on flooding algorithm
        Parameter:
        -------
            Sensor: list of 3 integers. Represents distance from next wall perceived from left, front and right sensors
            Target: list of tuples. Represents a set of maze boxes that robot is requested to reach in this phase.
                Note that the list may be different from maze center while in exploration mode.
                If the list has more than one box, reaching any box will satisfy the target
            Reset: Boolean. If True, flooding calculation need to restart from target (becouse the target was changed)
                When False, an optimized version of the flooding algorithm may be executed, saving part of flooding calculation
            mx_step: integer. Max allowed number of boxes to be traversed in the move

        Return:
        -------
        rotation:
            robot rotation (if any), as a number: 0 for no rotation, +90 for a 90-degree rotation clockwise,
            and -90 for a 90-degree rotation counterclockwise.
        movement:
            number of squares the robot is supposed to move. Movement is backwards if the number is negative.

        Special return cases:
        -------
        0,0: means robot location is within the target, so no move is needed and calling function must take actions
        """

        #target has changed, so we need to have a fresh flooding start
        if reset:
            self.flood = flooding(self.maze_dim, self.walls, target)

        #step 1: examine if sensors discovered new walls and update walls accordingly
        walled_boxes= self.__update_walls(sensors)

        #step 2: Recalculate flooding if we have new walls. Only flood values impacted by the change are recalculated
        if len(walled_boxes)>0:
            #we have new walls
            start=10000000
            for b in walled_boxes:
                start=min(start, self.flood[b])
            self.flood = adjust_flooding(self.maze_dim, self.walls,self.flood,start)
            #self.flood = flooding(self.maze_dim, self.walls,target)

        #step 3: given adjusted flooding, calculate best move

        #step 3.1: No move if we are within the target (flooding distance==0)
        if self.flood[tuple(self.location)]==0:
            return 0, 0

        # step 3.2: search beast reachable distance from target (i.e. lowest flooding value among reachable boxes)
        # note that a number of equivalent options are possible, all added to 'possibilities' list
        next_distance=self.flood[tuple(self.location)]
        possibilities=[]
        for d in directions:
            new_dist,steps=self.__explore(d,max_moves)
            if (new_dist<next_distance):
                possibilities=[(new_dist,steps,d)]
                next_distance = new_dist
            elif (new_dist==next_distance and steps>0):
                possibilities.append((new_dist,steps,d))

        # step 3.3: Randomly take a decision among 'possibilities' and calculate return values
        p=np.random.randint(0, len(possibilities))
        rotation=rotations[self.heading][possibilities[p][2]]
        movement = min(possibilities[p][1],mx_step)
        if rotation==180:
            rotation=0
            movement=-1
        # step 3.4: adjust robot location and return move
        self.__move_robot(rotation, movement)
        return rotation, movement


    def __explore(self,  direction, mx_step):
        """
        Calculate best number of step to move in a given direction
        Parameter:
        -------
            direction: a value in ['u','d','l','r'] (up, down, left, right)
            mx_step: integer. Max allowed number of boxes to be traversed in a move

        Return:
        -------
        new_dist:
            Lowest flooding value that might be reached moving in the given direction
        steps:
            Number of boxes to move in order to reach lowest flooding value
        """

        start=tuple(self.location)
        new_dist=self.flood[start]
        steps=0
        for s in range(mx_step+1):
            box=(start[0]+dir_move[direction][0]*s,start[1]+dir_move[direction][1]*s)
            if self.flood[box]<=new_dist:
                new_dist=self.flood[box]
                steps=s
            if (self.walls[box] & wall_mask[direction]) <= 0:
                break
        return new_dist, steps


    def __move_robot(self, rotation, movement):
        """
        Updates location and heading in robot state, given actual location and heading, rotation and movement
        Parameter:
        -------
            rotation: integer in  [90,0,-90,180]
            movement: integer. Number of maze boxes to be passed
        Return: None
        """
        if rotation in [0,180]:
            new_heading=self.heading
        else:
            new_heading=next_heading[self.heading][rotation]
        new_loc=[self.location[0]+dir_move[new_heading][0]*movement, self.location[1]+dir_move[new_heading][1]*movement]
        self.heading=new_heading
        self.location[0]=new_loc[0]
        self.location[1]=new_loc[1]


    def init_exp_target(self):
        """
        The method is used to initialize the robot informations that will drive maze exploration in run 0, specifically
         while in 'exploration_from_target' mode(some more info below)

        Parameter: None (the method uses the global variable 'exploring_paths' as input
        -------
        Return:
            exp_target: list of tuples, representing the next target for exploration (while in 'exploration_from_target' mode)
            exp_step: integer, representing the ongoing step in the exploring path sequence

        Additional explanation:
        ----------------------
        Exploration mode is a mean available to the robot to continue maze exploration after reaching the maze center in run0.
        Exploration has been implemented as a sequence of exploration step, grouped in the global dictionary 'exploring_paths'.
        The single exploration path followed by the robot is defined by the class attribute 'exp_path', using as a key for the
        global dictionary. As the global dictionary is not aware of maze dimension, the path is defined as a list of strings,
        to be interpreted as follows:

        'bl': bottom left corner of the maze (box 0,0)
        'br': bottom right corner of the maze (n,0)
        'tl': top left corner of the maze (0,n)
        'tr': top right corner of the maze (n,n)
        'cc': any of the 4 boxes in the maze center (as defined by requested target)

        As an example the exploring path [['br', 'tl', 'tr'], ['bl'],['cc]] would be interpreted by the robot as:
        1) Start from 0,0 and reach the center (this is a predefined step independent by chosen path)
        2) ['br', 'tl', 'tr']: Reach any maze corner different by (0,0)
        3) ['bl']: Then reach bottom left corner (0,0)
        4) ['cc']: Then reach any of the 4 box in the maze center
        5) Reset and start run 1
        """

        exp_target=[]
        exp_step=0
        for step in exploring_paths[self.exp_path]:
            s=[]
            for box in step:
                if box=='cc':
                    for center_box in self.target:
                        s.append(center_box)
                elif box == 'bl':
                    s.append((0,0))
                elif box == 'br':
                    s.append((self.maze_dim-1, 0))
                elif box == 'tl':
                    s.append((0, self.maze_dim-1))
                elif box == 'tr':
                    s.append((self.maze_dim-1, self.maze_dim-1))
            exp_target.append(s)
        return exp_target, exp_step


    def init_known_walls(self):
        """
        Initialize known_hor and known_vert arrays using available information about maze (perimetral wall, 0,0 box)
        Parameter: None
        Self changes
        -------
            known_vert and known_hor are created and initialized
        Return: None
        """
        n=self.maze_dim
        known_vert=np.full((n+1,n),0)
        known_hor=np.full((n,n+1),0)
        known_vert[0,:]=1
        known_vert[n,:]=1
        known_hor[:,0]=1
        known_hor[:,n]=1
        known_vert[1,0]=1
        self.known_hor=known_hor
        self.known_vert =known_vert


    def init_walls(self):
        """
        Initialize 'walls' array using available information about maze (perimetral wall, 0,0 box)
        Parameter: None
        Self changes
        -------
            walls is created and initialized
        Return: None
        """
        n=self.maze_dim
        walls=np.full((n,n),15)
        walls[0]=7
        walls[n-1]=13
        walls[:,0]=11
        walls[:,n-1]=14
        walls[0,0]=1
        walls[1,0]=3
        walls[n-1,0]=9
        walls[0,n-1]=6
        walls[n-1,n-1]=12
        self.walls=walls


    def __update_walls(self,sensing):
        """
        For each movement phase, the robot uses an imprecise representation of the maze (in the 'walls' array), given
        by the history of informations taken during exploration. This method provides information update before a move
        given the information taken by the three sensors.

        Parameter:
        -------
            sensing: a list of three integers, representing distance from next wall measured by left, front and right
            sensor (in this order)
        Self changes
        -------
            walls, known_vert and known_hor are updated
        Return:
            walled boxes: a list of tuples, representing the set of boxes for which a new wall has been discovered
            (each new wall imply addition of 2 boxes, so len of walled boxes is in [0,2,4,6])
        """
        n=self.maze_dim
        walled_boxes=[]
        col=self.location[0]
        row=self.location[1]
        for i,d in enumerate(dir_sensors[self.heading]):
            if d == 'r':
                self.known_vert[col+1:col+1+sensing[i]+1,row]=1
            elif d == 'l':
                self.known_vert[col-sensing[i]:col+1,row]=1
            elif d == 'u':
                self.known_hor[col,row+1:row+1+sensing[i]+1]=1
            elif d == 'd':
                self.known_hor[col,row-sensing[i]:row+1]=1

            walled_box=(col+dir_move[d][0]*sensing[i],row+dir_move[d][1]*sensing[i])
            if (self.walls[walled_box] & wall_mask[d])>0:
                self.walls[walled_box]= (self.walls[walled_box] - wall_mask[d])
                adj_box = (walled_box[0] + dir_move[d][0], walled_box[1] + dir_move[d][1])
                self.walls[adj_box] = (self.walls[adj_box] - wall_mask_rev[d])
                for m in range(max_moves,-max_moves,-1):
                    box = (walled_box[0]+dir_move[d][0]*m,walled_box[1]+dir_move[d][1]*m)
                    if max(box)<n and min(box)>=0:
                        walled_boxes.append(box)
        return walled_boxes

def flooding_old(n,walls,target):
    """
    Maze navigation toward current target is based on the 'flooding' algorithm.
    Basically, the algorithm starts assigning a 0 value to all boxes that are part of the current target, meaning that
    the distance from target for the box is obviously zero.
    Step 1 of the algorithm consists of marking with '1' those boxes that are still unmarked and are directly
    adjacent (without walls) to a box marked with 0
    This is repeated till all boxes are marked (assuming it exists a path from the target to each box in the maze)

    Parameter:
    -------
        walls: nxn array of integers, giving the current knowledge about walls in the maze.
            walls[i,j] provides a 4 bit based representation of the walls in maze box [i,j], 1 bit mean missing wall.
            0, 15: 4 walls and no walls respectively, 1,2,4,8: box exit only on top, right, bottom and left respectively
            In all case we have no info about existence of a wall, no wall is assumed
        target: list of tuples, representing the target in the maze that the robot is requested to reach

    Return:
    -------
        flood: nxn array, where flood[i,j] represent the distance (in number of boxes) from box [i,j] to the nearest box
        in target.
    """

    #flood is initialized with value -1 (no box explored)
    flood=np.full((n,n),-1)

    #target boxes are initialized with value 0 (0 distance from target)
    for b in target:
        flood[b]=0

    iter=0
    #current is the list of boxes explored and valued in the last iteration (those with highest distance among those explored)
    current=target
    while len(current)>0:
        iter+=1
        new=[]
        #new is the list of boxes that directly communicate with a box in current and were not yet explored
        for b in current:
            for d in directions:
                if (walls[b] & wall_mask[d]) > 0:
                    new_b=(b[0]+dir_move[d][0],b[1]+dir_move[d][1])
                    if (flood[new_b]<0):
                        flood[new_b]=flood[b]+1
                        new.append(new_b)
        current=new
        if iter>n*n:
            raise Exception('Flooding function not working properly')
    if flood[0,0]<0:
        print_maze_info(flood, n)
        raise Exception('Check Code, apparently no path from start to center')
    if flood.min()<0:
        print("Warning: part of the Maze is not reachable from current target")
        print_maze_info(walls,n)
        print('---------------------------------------------------')
        print_maze_info(flood,n)
    return flood


def flooding(n,walls,target):
    """
    Maze navigation toward current target is based on the 'flooding' algorithm.
    Basically, the algorithm starts assigning a 0 value to all boxes that are part of the current target, meaning that
    the distance from target for the box is obviously zero.
    Step 1 of the algorithm consists of marking with '1' those boxes that are still unmarked and are directly
    adjacent (without walls) to a box marked with 0
    This is repeated till all boxes are marked (assuming it exists a path from the target to each box in the maze)

    Parameter:
    -------
        walls: nxn array of integers, giving the current knowledge about walls in the maze.
            walls[i,j] provides a 4 bit based representation of the walls in maze box [i,j], 1 bit mean missing wall.
            0, 15: 4 walls and no walls respectively, 1,2,4,8: box exit only on top, right, bottom and left respectively
            In all case we have no info about existence of a wall, no wall is assumed
        target: list of tuples, representing the target in the maze that the robot is requested to reach

    Return:
    -------
        flood: nxn array, where flood[i,j] represent the distance (in number of boxes) from box [i,j] to the nearest box
        in target.
    """

    #flood is initialized with value -1 (no box explored)
    flood=np.full((n,n),-1)

    #target boxes are initialized with value 0 (0 distance from target)
    for b in target:
        flood[b]=0

    iter=0
    #current is the list of boxes explored and valued in the last iteration (those with highest distance among those explored)
    current=target
    while len(current)>0:
        iter+=1
        new=[]
        #new is the list of boxes that directly communicate with a box in current and were not yet explored
        for b in current:
            for d in directions:
                for m in range(max_moves):
                    bc=(b[0]+dir_move[d][0]*m,b[1]+dir_move[d][1]*m)
                    if (walls[bc] & wall_mask[d]) > 0:
                        new_b=(bc[0]+dir_move[d][0],bc[1]+dir_move[d][1])
                        if (flood[new_b]<0):
                            flood[new_b]=flood[b]+1
                            new.append(new_b)
                    else:
                        break
        current=new
        if iter>n*n:
            raise Exception('Flooding function not working properly')
    if flood[0,0]<0:
        print_maze_info(flood, n)
        raise Exception('Check Code, apparently no path from start to center')
    if flood.min()<0:
        print("Warning: part of the Maze is not reachable from current target")
        print_maze_info(walls,n)
        print('---------------------------------------------------')
        print_maze_info(flood,n)
    return flood


def adjust_flooding(n,walls,flood,start):
    """
    A move of the robot causes the repositioning of the sensors and possibly the discovery of new walls.
    The existence of new walls clearly impact the flood array, that must be recalculated, possibly using the
    flooding function. However, if a wall has been added between a box with flood distance 'd_1' and a box with
    flood distance 'd_2' we can safely assume that:
    - All boxes with current flooding distance lower than min (d_1,d_2) are not affected by the change
    - boxes with flooding distance higher than that must be recalculated
    These allow limiting re-calculation to a subset of the maze boxes, possibly adding some efficiency to the code

    Parameter:
    -------
        walls: nxn array of integers, giving the current knowledge about walls in the maze.
            walls[i,j] provides a 4 bit based representation of the walls in maze box [i,j], 1 bit mean missing wall.
            0, 15: 4 walls and no walls respectively, 1,2,4,8: box exit only on top, right, bottom and left respectively
            In all case we have no info about existence of a wall, no wall is assumed
        flood: nxn array of integers, representing flooding distances to be updated
        start: the minimum flood distance from which to start recalculation

    Return:
    -------
        flood: nxn array, where flood[i,j] represent the distance (in number of boxes) from box [i,j] to the nearest box
        in target.
    """

    #mark as 'unvisited' (flood=-1) all boxes that have a distance greater tha start
    to_reset=np.where(flood>start)
    for i in zip(to_reset[0],to_reset[1]):
        flood[i]=-1
    #start flooding algoritm from boxes with flood distance exactly equal to start
    to_current=np.where(flood==start)
    current=[]
    for i in zip(to_current[0],to_current[1]):
        current.append(i)

    #iterate among unvisited as in the normal flooding function
    iter=0
    while len(current)>0:
        iter+=1
        new=[]
        for b in current:
            for d in directions:
                for m in range(max_moves):
                    bc=(b[0]+dir_move[d][0]*m,b[1]+dir_move[d][1]*m)
                    if (walls[bc] & wall_mask[d]) > 0:
                        new_b=(bc[0]+dir_move[d][0],bc[1]+dir_move[d][1])
                        if (flood[new_b]<0):
                            flood[new_b]=flood[b]+1
                            new.append(new_b)
                    else:
                        break
        current=new
        if iter>=n*n:
            raise Exception('Flooding function not working properly')
    if flood[0,0]<0:
        print_maze_info(flood, n)
        raise Exception('Check Code, apparently no path from start to center')
    if flood.min()<0:
        print("Warning: part of the Maze is not reachable from maze center")
        print_maze_info(walls,n)
        print('---------------------------------------------------')
        print_maze_info(flood,n)
    return flood



def print_maze_info(info,n):
    """
      print an info about the maze (flood or walls) leaving entry (0,0) at the bottom left of the output
    """
    for r in range(n,0,-1):
        print(info[:,r-1])


def hit_target(location,target):
    """
      true if location (a tuple) is within target (a list of tuple)
    """
    for t in target:
        if tuple(location) == t: return True
    return False


def countwalls(walls,n):
    """
      count and return the number of walls in the maze, according to walls parameter
    """
    n_walls=n*2
    for i in range(n):
        for j in range(n):
            if (walls[i,j] & wall_mask['d'])==0: n_walls+=1
            if (walls[i,j] & wall_mask['r'])==0: n_walls+=1
    return n_walls

def evaluate(maze,exp_path,max_it):
    """
    Allocate a new class robot object and evaluate its performance in the input maze, given the key for exploration path to be followed
    Parameter:
     -------
         maze: nxn array of integers, describing maze walls.
         exp_path: key used to interrogate the exponential path global dictionary
         max_it: max number of allowed moves

     Return:
     -------
        moves: 2-integer lists, giving steps in run 0 and run 1
        known: 2-integer lists, the number of passages known to the robot at the end of each run
        score: float, yhe overall score
        rob.walls: the walls knowledge acquired from the robot at the end of run 1
     """
    rob = Robot(maze.dim)
    rob.exp_path=exp_path
    rob.exp_target,rob.exp_step = rob.init_exp_target()
    moves = [0, 0]
    known =[0,0]
    run = 0
    nwalls=maze.dim*(maze.dim+1)*2
    for j in range(max_it):
        sensing = [maze.dist_to_wall(rob.location, heading) for heading in dir_sensors[rob.heading]]
        rotation, movement = rob.next_move(sensing)
        moves[run]+=1
        if rotation=='Reset':
            rob.location=[0,0]
            rob.heading='u'
            known[0] = rob.known_hor.sum() + rob.known_vert.sum()
            run = 1
        if run==1 and hit_target(rob.location,rob.target):
            score=moves[0] / 30 + moves[1]
            known[1] = rob.known_hor.sum() + rob.known_vert.sum()
            #print(i,rob.exp_path,' Run 0: {} steps, {} known.  Run 1: {} steps, known: {}/{}. SCORE: {}'.format(moves[0], known[0],moves[1], known[1],nwalls, score))
            break
    if (j>=max_it):
        print("Target not reached!!")
        exit()
    #print_maze_info(rob.walls,rob.maze_dim)
    return(moves,known,score,rob.walls)


if __name__ == '__main__':
    #the loop creates the dataframe for performance analysis
    np.random.seed(10)
    directory = r'.\validrandommaze'
    outcome=[]
    for entry in os.scandir(directory):
        if entry.path.endswith(".txt"):
            print(entry.path)
            maze=Maze(entry.path)
            n=maze.dim
            hn=int(maze.dim/2)
            target = [(hn-1,hn-1),(hn-1,hn),(hn,hn-1),(hn,hn)]
            flood=flooding(n,maze.walls,target)
            for exp_path in exploring_paths.keys():
                for i in range(10):
                    d={}
                    moves, known, score, walls=evaluate(maze, exp_path,10000)
                    d['MazeName']=entry.path
                    d['MazeSize']=maze.dim
                    d['MazeAttempt']=int(entry.path.split('_')[2])
                    d['MazeNPossibleWalls']=maze.dim*(maze.dim+1)*2
                    d['MazeNWalls']=countwalls(maze.walls,maze.dim)
                    d['MazeShortestPath']=flood[0,0]
                    d['RobotExpMode']=exp_path
                    d['RobotIteration']=i
                    d['RobotRun0Moves']=moves[0]
                    d['RobotRun1Moves']=moves[1]
                    d['RobotRun0Known']=known[0]
                    d['RobotRun1Known']=known[1]
                    d['RobotKnownWalls']=countwalls(walls,maze.dim)
                    d['RobotPercWallsDisc']=d['RobotKnownWalls']/d['MazeNWalls']
                    d['RobotPercMazeDiscRun0']=d['RobotRun0Known']/d['MazeNPossibleWalls']
                    d['RobotScore']=moves[0]/30+moves[1]
                    outcome.append(d)
    pd.DataFrame(outcome).to_csv('RobotEvaluationDataTest.csv')

