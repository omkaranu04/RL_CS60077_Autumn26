Assignment 2: Hidden Reward Grid - Maximum Reward Path Using Q-Learning
A tabular Q-Learning agent that learns to travel from 'S' to 'G' on a grid whose rewards are hidden until visited

Files:
grid.py             -> The Grid environment
agent.py            -> The Q-Learning agent
utils.py            -> Training loop, printing/plotting helpers, and interactive-prompt helpers
main.py             -> The main script to run the program. RUN THIS FILE TO EXECUTE THE PROGRAM
test_environment.py -> Unit tests for the Grid environment
test_agent.py       -> Unit tests for the Q-Learning agent
test_cases.py       -> Scenario based end-to-end tests for the Agent and the Environment

Requirements:
    Python 3.10+
    numpy
    matplotlib
    pytest

How to Run?
    python main.py            -> interactive mode (prompts for grid size, start/goal,
                                  hidden reward values, and hyperparameters)
    python main.py --demo     -> runs directly with the PDF's example grid
    Flags: --seed N, --episodes N, --eps-decay X, --output-dir DIR
    e.g. python main.py --demo --episodes 1500 --eps-decay 0.995
    Results/plots are saved to output/ (created next to main.py, overwritten each run)

Use of the test_cases.py file:
    python test_cases.py
    Runs predefined scenarios end-to-end and prints a PASS/FAIL summary

Use of test_environment.py and test_agent.py files:
    pytest test_environment.py   -> tests the Grid environment
    pytest test_agent.py         -> tests the Q-Learning agent
