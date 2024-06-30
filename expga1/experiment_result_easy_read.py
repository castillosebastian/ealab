import json

# Custom encoder to format lists in a single line
class SingleLineListEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, list):
            return '[' + ', '.join(map(str, obj)) + ']'
        return super(SingleLineListEncoder, self).encode(obj)

def reformat_json(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    # Save the modified JSON file with single-line lists
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4, cls=SingleLineListEncoder)


# Example usage
input_file = 'expga1/experiments.json'  # Replace with your input file path
output_file = 'expga1/experiments_horizontal.json'  # Replace with your desired output file path

reformat_json(input_file, output_file)
