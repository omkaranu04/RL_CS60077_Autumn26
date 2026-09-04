Assignment 1: Grid World Navigation Using Q-Learning
A tabular Q-Learning agent that learns to navigate a grid worlf from 'S' to 'G' while avoiding 'X'

Files:
qlearning.py    -> The Grid environment, the Agent, and the Q-Learning algorithm
main.py         -> The main script to run the program. RUN THIS FILE TO EXECUTE THE PROGRAM
test_cases.py   -> Test cases for the Q-Learning agent (different grid sizes, start/goal positions, obstacles, and hyperparameters)

Requirements:
    Python 3.10+
    numpy
    matplotlib

How to Run?
    Run the main.py file - python main.py (you will be prompted to enter the details of th environment and the hyperparameters)
    All plots are saved to the output/ folder created in the samed directory as main.py (they get refreshed for every run)

Use of the test_cases.py file:
    Run the test_cases.py file - python test_cases.py
    Test cases are written within the file, they will be run and verified automatically. Results printed on the terminal 