Assignment 3: Two-Agent Tic-Tac-Toe Using Q-Learning
Two independent Q-Learning agents (Agent X and Agent O) that learn to play Tic-Tac-Toe against
each other purely through self-play, converging toward strong play (ideally forcing draws)

Files:
board.py            -> The Tic-Tac-Toe board environment
agent.py            -> The Q-Learning agent
utils.py            -> Self-play training loop, printing/plotting helpers, and interactive-prompt helpers
main.py             -> The main script to run the program. RUN THIS FILE TO EXECUTE THE PROGRAM
test_environment.py -> Unit tests for the Board environment
test_agent.py       -> Unit tests for the Q-Learning agent
test_cases.py       -> Scenario based end-to-end tests for the agents and the environment

Requirements:
    Python 3.10+
    numpy
    matplotlib
    pytest

How to Run?
    python main.py            -> interactive mode (prompts for board size and hyperparameters)
    python main.py --demo     -> runs directly with the classic 3x3 setup
    Flags: --seed N, --episodes N, --eps-decay X, --output-dir DIR
    e.g. python main.py --demo --episodes 30000 --eps-decay 0.999
    Results/plots are saved to output/ (created next to main.py, overwritten each run)

Use of the test_cases.py file:
    python test_cases.py
    Runs predefined scenarios end-to-end and prints a PASS/FAIL summary

Use of test_environment.py and test_agent.py files:
    pytest test_environment.py   -> tests the Board environment
    pytest test_agent.py         -> tests the Q-Learning agent
