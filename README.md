# Detection of Manipulated Pricing in Smart Energy CPS Scheduling
## Description
The implementation of a technique to model training data and compute the labels for all testing data regarding the energy consumption by a small community (5 people), with each user having a set of tasks to be performed during the day.

## How to Run:
* “make setup” - Runs a “requirements.txt” file which installs the required modules to run the various file.
* “make run” - Runs “classify.py” which classifies all the data pieces as normal (0) or abnormal (1).
* “make print” - Prints the predictions (0 or 1) in an output file "TestingResults.txt".
* “make compare” - Runs different classification methods and outputs the accuracy of each model.
* “make schedule” - Plots the schedules for each of the normal datasets in the folder called "all_plots".