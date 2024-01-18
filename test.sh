#!/bin/bash

start_time=$(date +%s.%N)  # Record the start time

for i in {1..300}; do
    python test-realtime.py
    
    if [ $i -eq 100 ]; then
        echo "Iteration $i completed"
        end_time_100=$(date +%s.%N)    # Record the end time
        elapsed_time=$(echo "$end_time_100 - $start_time" | bc)  # Calculate the elapsed time
        echo "Total time for 100 iterations: $elapsed_time seconds"
        continue
    fi

    echo "Iteration $i completed"
done

end_time=$(date +%s.%N)    # Record the end time
elapsed_time=$(echo "$end_time - $start_time" | bc)  # Calculate the elapsed time
echo "Total time for all iterations: $elapsed_time seconds"
