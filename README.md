# UdacityCapstoneProject

## Project description
Repository for delivery of Capstone Project for data Science Udacity Nanodegree

The project is addressing one of the alternatives proposed as a capstone project for Data Science Udacity Nanodegree.
Specifically the exercice asked to code a 'robot' python class, capable to guide a virtual robot moving into an unknown maze, from a given starting point toward the maze center.
The code should provide a digital abstraction of robot running in 'micromouse IEEE competition'   

## Repository content
The outcome of the exercise is collected in a number of deliverables, including some python modules, one jupiter notebook, a trial set of mazes, a csv coded dataframe and a pdf write up. A  write up of the project similar to pdf file was also posted as a Mudim blog at https://roberto-paglino.medium.com/discover-and-and-navigate-a-virtual-maze-9297abba2db8. Details on repository content are summerized below:

### Python modules:
- robot.py: (the main deliverable) contains the requested 'Robot' class, with method and attributes that enable robot movement and maze discovery
- mycreatemaze.py: contains an algorithm developed to create a set of random mazes intended for robot performance testing
- maze.py, tester.py, showmaze.py: additional project input, intended respectively to provide the 'maze' class, to test robot navigation in a maze, and to provide a visula picture of a specific maze 

### maze files (.txt):
- The project definition include some general maze specification and a set of three mazes proposed fpr software validation. These are included in the repository as 'test_maze_01(2,3).txt'.
- in addition, a trial set of 240 random mazes was created for robot performance evaluation. These are coded according to project specification and are all included in folder 'validrandommaze' 

### csv dataset:
- Folder 'validrandommazes' also include a csv-coded pandas dataset, 'RobotEvaluationDataTest.csv'. Each row in the dataset include information about a specific run of the robot on a specific maze. Each rows has info about the used maze (filename, size, etc), robot configuration (the exploration path used) and robot performance in that configuration and on that maze    
  
### Jupiter notebook:
- TrialMazeAnalysis.ipynb: A notebook including an analysis of 'RobotEvaluationDataTest.csv' 

### pdf write up:
- report.pdf: As anticipated, the actual report of the project was posted as a medium contribution. The same content (more or less) was also saved as pdf and included in the repository. 

### Folder organization
All content is included in the main repository, with the exception of trial mazes and Jupiter notebook (all included in 'validfandommaze' folder) 

## Prerequisites

The software uses the following libraries: 
- numpy
- pandas
- os
- seaborn (Notebook only)
- matplotlib  (Notebook only)


## Usage

The code was developed and tested under PyCharm environment

Robot software can be tested from the command line with a command like the following: python tester.py mazename.txt.

Maze visualization can be obtained similarly using python showmaze.py mazename.txt. 

Running Robot.py as main will produce a new version of the 'RobotEvaluationDataTest.csv' file. Specifically, the code will scan mazes within folder '.\validrandommaze'; 
for each maze in the folder, the code will go through 8 different robot working mode, and for each 'maze, mode' pair it will run 10 different maze exploration execution (for a total of 240 x 8 x 10 = 19200 executions).

When called as a class, (e.g. from tester.py) robot is configured to use the working mode that proved to be more performant on the trial maze set.

Notebook expect to start in the folder containing the 'valirandommaze' subfolder  

## Acknowledgement

All code provided in robot.py and randommaze.py modules was originally produced by me for this project (with the exception of what was part of the robot.py version provided as project input). Some data structures used in robot.py, specifically some global variables and the ‘walls’ attribute organization, were derived by the ‘maze.py’ modules that was also provided as part of the project input.

Flooding algorithm and its usage in an unknown maze context was derived from https://medium.com/@minikiraniamayadharmasiri/micromouse-from-scratch-algorithm-maze-traversal-shortest-path-floodfill-741242e8510

## License

Free to use

## Contact

roberto.paglino@gmail.com

## Project Link: 

https://github.com/rpaglin/UdacityCapstoneProject
