# Assignment 1: Grid World

DO NOT FORK THIS REPO
-----
Clone it and work on it locally.


Setting Up
----

Here is a recommended way to set up your virtual environment:

1. Install [uv](https://github.com/astral-sh/uv), a package manager like pip.
2. Run `uv sync`
3. Run `source .venv/bin/activate`

We'll mostly work with PyGame for all assignments. 

Task
----
The task is to find paths from the start (yellow node) to the goal (orange node). Once you load up the program (You'll
see how in the `Usage` section below) and press `enter` you will see what that means. In class I briefly explained the
meaning of the different colors of the nodes (green is "grass" that incurs high cost, blue is "puddle" that the agent
can not pass through). Check slides, lecture recording, and read the code to figure things out. If you are stuck, feel
free to discuss on slack or attend office hours.

The code for DFS is **partially** given to make it easy for you to understand the code. But it is intentionally buggy.
Before you start implementing the other algorithms, fix DFS first.

Implement the following search strategies in `ai.py`:

- DFS (buggy)
- BFS
- Uniform Cost Search
- A\* Search using Manhattan Distance as the heuristic

You can find the function definitions in the file. Feel free to add more auxiliary functions if needed. You can use
other **standard Python libraries** such as math etc. but they are not really needed. Do not use libraries not standard
to Python (i.e. numpy, torch); if you are not sure whether a library is ok to use or not, ask in the slack group.

Usage
----


Run `python main.py` to open the grid world window. 
Press a number, then `enter`, to see how each method runs.
As you will see, DFS is buggy; you need to fix it.
Pressing 2, 3, or 4 should respectively run BFS, UCS, A\* in a similar way, which you will implement (since right now it does nothing).

The `tests` file contains all test maps.
If you want to load in the maps as test cases (
`python main.py -l [test case number]`), we have provided a few fun cases for you to play with:

0. CSE ~~11~~ 150b style (homage to Rick Ord)
1. "It's a-me, Mario!"

The remaining test cases are randomly generated. To automatically test on all given tests, run `python main.py -t` which
autogrades the algorithms with respect to the correct optimal costs in the test maps above.

Submission
----
Only submit the `ai.py` file on Gradescope for grading.

If you have changed other files, make sure that your implementation works properly with the unchanged given files.

Due date
-----
Apr-12 11:59pm Pacific Time.

Grading
-----
All test cases are given in `tests`.

- Full (5 points): Passes all given tests.
- Almost (3 points): Fails some given tests.
- Nothing (0 point): No attempt.

Late Policy: -1 point every late day (0 is floor)
