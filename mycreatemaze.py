import numpy as np

directions = ['u','r','l','d']
dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
               'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}
wall_mask = {'u': 1, 'r': 2, 'd': 4, 'l': 8,}

wall_mask_rev= {'u': 4, 'r': 8, 'd': 1, 'l': 2,
               'up': 4, 'right': 8, 'down': 1, 'left': 2}


def flooding(n,walls,target):
    """
    Calculate weights for the 'flooding' algorithm.
    Used in this module to check full maze reachability after addition of a new wall.

    Parameter:
    -------
        n: size of the maze
        walls: nxn array of integers, giving knowledge about walls in the maze.
        target: list of tuples, representing the target in the maze that the robot is requested to reach

    Return:
    -------
        An integer value. A value lower than zero indicates to the caller that some box was not reached
        by the flooding and therefore the added wall must be removed
    """

    flood=np.full((n,n),-1)
    check={'u':1, 'r':2, 'd':4, 'l':8}
    current=target
    for b in current:
        flood[b]=0
    iter=0
    while len(current)>0:
        iter+=1
        new=[]
        for b in current:
            for d in check.keys():
                if (walls[b] & check[d]) > 0:
                    new_b=(b[0]+dir_move[d][0],b[1]+dir_move[d][1])
                    if (flood[new_b]<0):
                        flood[new_b]=flood[b]+1
                        new.append(new_b)
        current=new
        if iter>=n*n:
            raise Exception('Flooding function not working properly')
    return flood.min()



def init_walls(n):
    """
    Initialize the array containing the maze walls.

    Parameter:
    -------
        n: size of the maze

    Return:
    -------
        walls nxn array
    """

    #set walls on the perimeter and on the right of (0,0) box
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

    #set walls on the perimeter of the maze centre
    c=int(n/2)
    walls[c-2,c]=13
    walls[c-2,c-1]=13
    walls[c-1,c+1]=11
    walls[c-1,c]=6
    walls[c-1,c-1]=3
    walls[c-1,c-2]=14
    walls[c,c+1]=11
    walls[c,c]=12
    walls[c,c-1]=9
    walls[c,c-2]=14
    walls[c+1,c]=7
    walls[c+1,c-1]=7

    #introduce a single open in the walls surrounding the mae center
    sel=np.random.randint(8)
    if sel==0:
        walls[c,c+1]=15
        walls[c,c]=13
    elif sel == 1:
        walls[c-1,c+1]=15
        walls[c-1,c]=7
    elif sel == 2:
        walls[c-2,c]=15
        walls[c-1,c]=14
    elif sel == 3:
        walls[c-2,c-1]=15
        walls[c-1,c-1]=11
    elif sel == 4:
        walls[c-1,c-1]=7
        walls[c-1,c-2]=15
    elif sel == 5:
        walls[c,c-1]=13
        walls[c,c-2]=15
    elif sel == 6:
        walls[c,c-1]=11
        walls[c+1,c-1]=15
    elif sel == 7:
        walls[c,c]=14
        walls[c+1,c]=15

    return walls


def add_one_wall(n,walls,target,max_attempt):
    """
    Try to add a wall to the maze, in a random position.
    As the new wall might be rejected if it create a break in the maze (some box get unreachable) the attempt is
    repeated a number of time, than stopped

    Parameter:
    -------
        n: size of the maze
        walls: the nxn array containing info about maze walls
        target: a list of tuples indicating the maze centre
        max_attempt: integer. The number of attempt done to create a new wall

    Return:
    -------
        walls: nxn array, unchanged or eventually with a new wall added
        wall_added: boolean, True if the function managed to add a new wall , False elsewhere
    """
    wall_added=False
    for i in range(max_attempt):
        #select a random box position and a random wall direction on that box
        box=(np.random.randint(0,n),np.random.randint(0,n))
        d=np.random.choice(directions)
        #if the wall is not already present, verify reachability adding that wall
        if (walls[box] & wall_mask[d])>0:
            next_box = (box[0]+dir_move[d][0],box[1]+dir_move[d][1])
            walls[box]= (walls[box] - wall_mask[d])
            walls[next_box] = (walls[next_box] - wall_mask_rev[d])
            # if the maze remain completely reachable (flooding >=0, keep the wall
            if(flooding(n,walls,target)>=0):
                wall_added=True
                return wall_added,walls
            # else undo wall addition
            else:
                walls[box] = (walls[box] + wall_mask[d])
                walls[next_box] = (walls[next_box] + wall_mask_rev[d])
                wall_added = False
    return wall_added, walls


def savemaze(n,walls,attempt,folder):
    """
    Save a maze as text file, following convention used in the input samples, so that can be properly read by showmaze.py
    Parameter:
    -------
        n: size of the maze
        walls: the nxn array containing info about maze walls
        attempt: integer. The number of attempt done to create a new wall. Inserted as part of maze file name
        folder: string, the folder where the maze file is saved

    Return:
    -------
        None
    """
    fname = folder + '/ran' + '_' + str(n) + '_' + str(attempt) + '_'
    for i in range(min(n,6)):
        fname=fname+str(walls[0,i])
    fname=fname+'.txt'
    f=open(fname,'w')
    f.write('%d\n' % n)
    for i in range(n):
        for j in range(n-1):
            f.write('%d,'% walls[i,j])
        f.write('%d\n'% walls[i,n-1])
    f.close()

def create_maze(n,attempt):
    """
    Create a random maze
    Parameter:
    -------
        n: size of the maze. Also included in maze file name
        attempt: integer. The number of attempt done to create a new wall. Also included in maze file name

    Return:
    -------
        None
    """
    target=[(int(n/2)-1,int(n/2)-1),(int(n/2)-1,int(n/2)),(int(n/2),int(n/2)-1),(int(n/2),int(n/2))]
    walls= init_walls(n)
    wall_added=True
    while wall_added:
        wall_added, walls=add_one_wall(n, walls, target, attempt)
    #assure that there are no internal walls in the central box
    walls[target[0]]=walls[target[0]] | wall_mask['u']
    walls[target[0]]=walls[target[0]] | wall_mask['r']
    walls[target[1]]=walls[target[1]] | wall_mask['d']
    walls[target[1]]=walls[target[1]] | wall_mask['r']
    walls[target[2]]=walls[target[2]] | wall_mask['u']
    walls[target[2]]=walls[target[2]] | wall_mask['l']
    walls[target[3]]=walls[target[3]] | wall_mask['d']
    walls[target[3]]=walls[target[3]] | wall_mask['l']
    return walls


if __name__ == '__main__':
    np.random.seed(10)
    for n in range(12,17,2):
        for attempt in [5,20,60,120]:
            for _ in range(3):
                walls = create_maze(n, attempt)
                savemaze(n, walls,attempt, 'randommaze')
