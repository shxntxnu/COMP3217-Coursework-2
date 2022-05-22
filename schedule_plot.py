from pulp import LpMinimize, LpProblem, lpSum, LpVariable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Read user, task, and energy information from Excel file
def readData():
    excel_file = pd.read_excel('COMP3217CW2Input.xlsx', sheet_name='User & Task ID')
    user_and_task = excel_file['User & Task ID'].tolist()
    ready_time = excel_file['Ready Time'].tolist()
    deadline = excel_file['Deadline'].tolist()
    max_energy_ph = excel_file['Maximum scheduled energy per hour'].tolist()
    energy_demand = excel_file['Energy Demand'].tolist()
    tasks = []
    users_and_tasks = []

    for k in range(len(ready_time)):
        task = [ready_time[k], deadline[k], max_energy_ph[k], energy_demand[k]]
        users_and_tasks.append(user_and_task[k])

        tasks.append(task)

    # Read output from data frame testing
    test_DF = pd.read_csv('TestingResults.txt', header=None)
    y_labels = test_DF[24].tolist()
    test_DF = test_DF.drop(24, axis=1)
    x_data = test_DF.values.tolist()

    return tasks, users_and_tasks, x_data, y_labels


# LP model creation function for the scheduling problem
def createLPModel(tasks, task_names):
    """LP model creation function for the scheduling problem"""

    # Variables for tasks
    task_vars = []
    c = []
    eq = []

    # Create LP problem model for Minimisation
    model = LpProblem(name="scheduling-problem", sense=LpMinimize)

    # Loop through list of tasks
    for ind, task in enumerate(tasks):
        n = task[1] - task[0] + 1
        temp_list = []
        # Loop from ready time to deadline for each task
        # Creates LP variables with given constraints and unique names
        for i in range(task[0], task[1] + 1):
            x = LpVariable(name=task_names[ind] + '_' + str(i), lowBound=0, upBound=task[2])
            temp_list.append(x)
        task_vars.append(temp_list)

    # Create objective function for price (to minimize) and add to the model
    for ind, task in enumerate(tasks):
        for var in task_vars[ind]:
            price = price_list[int(var.name.split('_')[2])]
            c.append(price * var)
    model += lpSum(c)

    # Add additional constraints to the model
    for ind, task in enumerate(tasks):
        temp_list = []
        for var in task_vars[ind]:
            temp_list.append(var)
        eq.append(temp_list)
        model += lpSum(temp_list) == task[3]

    # Return model to be solved in main function
    return model


# Plot hourly energy usage for community
def plot(model, count):
    global width
    hours = [str(x) for x in range(0, 24)]
    pos = np.arange(len(hours))
    users = ['user1', 'user2', 'user3', 'user4', 'user5']
    colours = ['black', 'red', 'blue', 'orange', 'green']
    plot_list = []
    to_plot = []

    # Create lists to plot usage
    for user in users:
        temp_list = []
        for hour in hours:
            hour_list_temp = []
            task_count = 0
            for var in model.variables():
                if user == var.name.split('_')[0] and str(hour) == var.name.split('_')[2]:
                    task_count += 1
                    hour_list_temp.append(var.value())
            temp_list.append(sum(hour_list_temp))
        plot_list.append(temp_list)
        width = 0.2

    # Shows schedule as grouped bar charts, sorted by user against hours
    plt.bar(pos - 0.5, plot_list[0], width, color=colours[0], edgecolor='black', bottom=0)
    plt.bar(pos - 0.3, plot_list[1], width, color=colours[1], edgecolor='black', bottom=0)
    plt.bar(pos - 0.1, plot_list[2], width, color=colours[2], edgecolor='black', bottom=0)
    plt.bar(pos + 0.1, plot_list[3], width, color=colours[3], edgecolor='black', bottom=0)
    plt.bar(pos + 0.3, plot_list[4], width, color=colours[4], edgecolor='black', bottom=0)

    plt.xticks(pos, hours)
    plt.xlabel('Hour')
    plt.ylabel('Energy Usage (kW)')
    plt.title('Energy Usage Per Hour For All Users\nDay %i' % count)
    plt.legend(users, loc=0)
    # plt.show()
    plt.savefig('all_plots\\0-normal\\' + str(count) + '.png')  # For Normal graph
    # plt.savefig('all_plots\\1-abnormal\\' + str(count) + '.png')  # For Abnormal graph
    plt.clf()

    return plot_list


tasks, task_names, x_data, y_labels = readData()

for ind, price_list in enumerate(x_data):
    # (Un)comment below to plot abnormal guideline pricing schedule
    # if y_labels[ind] == 1:
    #     # Solve returned LP model for scheduling solution of Abnormal values
    #     model = createLPModel(tasks, task_names)
    #     answer = model.solve()
    #     # Print LP model stats
    #     print(answer)
    #     # Plot hourly usage for scheduling solution
    #     plot(model, ind + 1)

    # (Un)comment below to plot normal guideline pricing schedule
    if y_labels[ind] == 0:
        # Solve returned LP model for scheduling solution of Normal values
        model = createLPModel(tasks, task_names)
        answer = model.solve()
        # Print LP model stats
        print(answer)
        # Plot hourly usage for scheduling solution
        plot(model, ind + 1)

print("\n Normal plots can be found in the all-plots/0-normal folder")
print("\n Abnormal plots can be found in the all-plots/1-abnormal folder")
