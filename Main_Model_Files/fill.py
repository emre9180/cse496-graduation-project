import json

# Load the JSON file
with open('updated_first_json.json', 'r') as file:
    data = json.load(file)

# Define the mapping for measure_id to real major and minor lengths
measure_map = {
    1: {'real_major': 3.7, 'real_minor': 2.8},
    2: {'real_major': 3.4, 'real_minor': 2.7},
    3: {'real_major': 4.3, 'real_minor': 3.4},
    4: {'real_major': 3.5, 'real_minor': 2.7},
    5: {'real_major': 3.3, 'real_minor': 3.1},
    6: {'real_major': 3.6, 'real_minor': 2.8},
    7: {'real_major': 2.6, 'real_minor': 2.6},
}

# Update the JSON data
for item in data:
    print(item)
    measure_id = item['measure_id']
    if measure_id in measure_map:
        item['real_major_axis_length'] = measure_map[measure_id]['real_major']
        item['real_minor_axis_length'] = measure_map[measure_id]['real_minor']

# Save the updated JSON to a new file
with open('update_first_json2.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Real major and minor axis lengths have been updated in the JSON file.")
