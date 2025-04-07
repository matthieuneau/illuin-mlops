start_time=$(date +%s.%N)
count=0

for file in data/texts/*.txt; do
    ((count++))

    if [ $count -gt 100 ]; then
        break
    fi

    file_content=$(cat "$file")
    curl --location 'http://0.0.0.0:8000/' \
        --header 'Content-Type: text/plain' \
        --data "\"$file_content\""
done
end_time=$(date +%s.%N)
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Total execution time: $execution_time seconds"
done
