import json

# Load the first JSON file
with open('first_try_ellipse_results.json', 'r') as file:
    first_json = json.load(file)

# Load the second JSON file
with open('updated_first_json2.json', 'r') as file:
    second_json = json.load(file)

# Find the maximum id in the first JSON
max_id = max(item['id'] for item in first_json)

# Update the id fields in the second JSON
for i, item in enumerate(second_json):
    item['id'] = max_id + 1 + i

# Add entries from the second JSON to the first JSON
first_json.extend(second_json)

# Save the merged JSON to a new file
with open('merged_json.json', 'w') as file:
    json.dump(first_json, file, indent=4)

print("Entries from the second JSON have been added to the first JSON with updated IDs.")
