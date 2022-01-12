# ------ SOMA Simple Program       ---  Version: All To One (Original) ----
# ------ Written by: Quoc Bao DIEP ---  Email: diepquocbao@gmail.com   ----
# -----------  See more details at the end of this file  ------------------
import numpy
import time
from List_of_CostFunctions import Schwefel as CostFunction

starttime = time.time()                                             # Start the timer
print('Hello! SOMA ATO is working, please wait... ')
dimension = 10                                                      # Number of dimensions of the problem
# -------------- Control Parameters of SOMA -------------------------------
Step, PRT, PathLength = 0.11, 0.1, 3                                # Assign values ​​to variables: Step, PRT, PathLength
PopSize, Max_Migration = 100, 100                                   # Assign values ​​to variables: PopSize, Max_Migration
# -------------- The domain (search space) --------------------------------
VarMin = -500 + numpy.zeros(dimension)                              # By hand, for example: VarMin = numpy.array([-500, -501,..., -500])
VarMax = 500 + numpy.zeros(dimension)                               # Lenght of VarMin and VarMax have to equal dimension
# %%%%%%%%%%%%%%      B E G I N    S O M A    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------- Create the initial Population -----------------------------
VarMin = numpy.repeat(VarMin.reshape(dimension, 1),PopSize,axis=1)  # Change VarMin from vector (1 x dimension) to matrix (dimension x PopSize)
VarMax = numpy.repeat(VarMax.reshape(dimension, 1),PopSize,axis=1)  # Change VarMax from vector (1 x dimension) to matrix (dimension x PopSize)
pop = VarMin + numpy.random.rand(dimension, PopSize) * (VarMax - VarMin) # Create the initial Population
fitness = CostFunction(pop)                                         # Evaluate the initial population
FEs = PopSize                                                       # Count the number of function evaluations
the_best_cost = min(fitness)                                        # Find the Global minimum fitness value
# ---------------- SOMA MIGRATIONS ----------------------------------------
Migration = 0                                                       # Assign values ​​to variables: Migration
while Migration < Max_Migration:                                    # Terminate when reaching Max_Migration / User can change to Max_FEs
    Migration = Migration + 1                                       # Increase Migration value
    idx = numpy.argmin(fitness)                                     # Find the index of minimum value in the fitness list
    leader = pop[:, idx].reshape(dimension, 1)                      # Get the Leader position (solution values) in the current population
    # ------------ movement of each individual ----------------------------
    for j in range(PopSize):                                        # Choose all individuals move toward the Leader
        indi_moving = pop[:, j].reshape(dimension, 1)               # Get the position (solution values) of the moving individual
        if j != idx:                                                # Don't move if it is itself
            offspring_path = numpy.empty([dimension, 0])            # Create an empty path of offspring
            for k in numpy.arange(Step, PathLength, Step):          # From Step to PathLength: jumping
                PRTVector = (numpy.random.rand(dimension,1)<PRT)*1  # If rand() < PRT, PRTVector = 1, else, 0
                offspring = indi_moving + (leader - indi_moving) * k * PRTVector # Jumping towards the Leader
                offspring_path = numpy.append(offspring_path, offspring, axis=1) # Store the jumping path
            size = numpy.shape(offspring_path)                      # How many offspring in the path
            # ------------ Check and put individuals inside the search range if it's outside
            for cl in range(size[1]):                               # From column
                for rw in range(dimension):                         # From row: Check
                    if offspring_path[rw][cl] < VarMin[rw][0] or offspring_path[rw][cl] > VarMax[rw][0]:  # if outside the search range
                        offspring_path[rw][cl] = VarMin[rw][0] + numpy.random.rand() * (VarMax[rw][0] - VarMin[rw][0]) # Randomly put it inside
            # ------------ Evaluate the offspring and Update -------------
            new_cost = CostFunction(offspring_path)                 # Evaluate the offspring
            FEs = FEs + size[1]                                     # Count the number of function evaluations
            min_new_cost = min(new_cost)                            # Find the minimum fitness value of new_cost
            idz = numpy.argmin(new_cost)                            # Find the index of minimum value in the new_cost list
            the_best_offspring = offspring_path[:, idz]             # Get the position values (solution values) of the min_new_cost
            # ----- Accepting: Place the best offspring into the current population
            if min_new_cost <= fitness[j]:                          # Compare min_new_cost with fitness value of the moving individual
                fitness[j] = min_new_cost                           # Replace the moving individual fitness value
                pop[:, j] = the_best_offspring                      # Replace the moving individual position (solution values)
                # ----- Update the global best value ----------------------
                if min_new_cost <= the_best_cost:                   # Compare Current minimum fitness with Global minimum fitness
                    the_best_cost = min_new_cost                    # Update Global minimun fitness value
                    the_best_value = the_best_offspring             # Update Global minimun position
# %%%%%%%%%%%%%%%%%%    E N D    S O M A     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
endtime = time.time()                                               # Stop the timer
caltime = endtime - starttime                                       # Caculate the processing time
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Show the information to User
print('Stop at Migration :  ', Migration)
print('The number of FEs :  ', FEs)
print('Processing time   :  ', caltime, '(s)')
print('The best cost     :  ', the_best_cost)
print('Solution values   :  ', the_best_value)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This algorithm is programmed according to the descriptions in the papers listed below:
# Link of paper: https://link.springer.com/chapter/10.1007/978-3-319-28161-2_1
# I. Zelinka and L. Jouni, "SOMA–self-organizing migrating algorithm mendel," in 6th International Conference on Soft Computing, Brno, Czech Republic, 2000.
# I. Zelinka, "SOMA–self-organizing migrating algorithm," in New optimization techniques in engineering. Springer, 2004, pp. 167–217.
# I. Zelinka, "SOMA–self-organizing migrating algorithm," in Self-Organizing Migrating Algorithm. Springer, 2016, pp. 3–49.