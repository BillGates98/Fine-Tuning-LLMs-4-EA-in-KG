for dataset in "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python3.8 ./linking.py --suffix $dataset >> ./outputs/output_linking.txt
done
