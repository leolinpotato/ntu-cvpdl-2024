#!/bin/bash

# Loop through confidence levels from 0.1 to 1.0 in steps of 0.1
for i in {1..10}; do
  # Calculate the confidence level (0.1, 0.2, ..., 1.0)
  conf=$(awk "BEGIN {printf \"%.1f\", $i / 10}")
  
  # Construct the output file name
  output_file="/data1/leolin/CVPDL/ntu-cvpdl-2024/hw3/tmp/predicted_visualization_200_stable_diffusion_w_prompt_${conf}.json"
  
  # Run the Python script with the current confidence level
  python select_conf.py --conf $conf --input_json '/data1/leolin/CVPDL/ntu-cvpdl-2024/hw3/predicted_visualization_200_stable_diffusion_w_prompt.json' --output_json "$output_file"

  # Print a message indicating the output file has been generated
  echo "Generated output: $output_file with conf=$conf"
done