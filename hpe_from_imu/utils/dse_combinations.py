# Write all combinations of design space analysis into CSV file (ID: Int, Combination: List ,Trained: bool)
import csv
from itertools import combinations, chain

lenght_min = 1
lenght_max = 10
sensor_ID = [(1,),(2,),(7,),(20,),(3,4),(5,6),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(21,22),(23,24)]

combs = []
for i in range(lenght_min, len(sensor_ID)):
    comb = list(combinations(sensor_ID, i))
    print("There are {} combinations with {} tuples".format(len(comb),i))
    combs.extend(comb)

# Process obtained combinations
comb_val = [] 
for index, combination in enumerate(combs):
    # Convert list of Tuples into list of Ints
    out = list(sum(combination, ()))
    out.extend([0])
    # Delet invalid combinations
    if len(out) > lenght_max:
        continue # Too many sensors -> out of scope
    elif((3 in out and 8 in out) or (3 in out and 14 in out) or (8 in out and 14 in out)):
        continue # Upper Leg
    elif(10 in out and 12 in out):
        continue # Upper Arm
    elif(16 in out and 21 in out):
        continue # Forearm
    elif(18 in out and 23 in out):
        continue # Lower Leg
    else:
        comb_val.append(out)

print("\nIn common, {} valid combinations have been found, with {} to {} sensors".format(len(comb_val),lenght_min, lenght_max))

# Prepare Columns
ID = [i for i in range(1,(len(comb_val))+1)]
trained = [False for i in range(1,(len(comb_val))+1)]
rows = zip(ID, comb_val, trained)

# write csv file ID, List of Sensors, trained 
file = open('dse_combinations.csv', 'w', newline ='')

with file:
	writer = csv.writer(file, delimiter=';')
	writer.writerows(rows)



