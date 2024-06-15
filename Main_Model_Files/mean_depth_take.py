import json

tup = [ []]

# Load the first JSON file
with open('ellipse_results.json', 'r') as file:
    first_json = json.load(file)

# Load the second JSON file
with open('depth_results.json', 'r') as file:
    second_json = json.load(file)

# Create a dictionary to map file_name to a list of mean_depth values
depth_dict = {}
for item in second_json:
    file_name = item['file_name']
    if file_name not in depth_dict:
        depth_dict[file_name] = []
    depth_dict[file_name].append((item['id'], item['mean_depth']))

# Sort the depth values by ID in ascending order
for key in depth_dict:
    depth_dict[key].sort(key=lambda x: x[0])

# Create a counter to keep track of the current index for each file_name in the depth dictionary
depth_index = {key: 0 for key in depth_dict}

# Fill mean_depth in the first JSON based on matching file_name and id order
for item in first_json:
    file_name = item['file_name']
    if file_name in depth_dict:
        if depth_index[file_name] < len(depth_dict[file_name]):
            item['mean_depth'] = depth_dict[file_name][depth_index[file_name]][1]
            depth_index[file_name] += 1

# Save the updated first JSON to a new file
with open('updated_first_json.json', 'w') as file:
    json.dump(first_json, file, indent=4)

print("Mean depth values have been updated in the first JSON file.")
