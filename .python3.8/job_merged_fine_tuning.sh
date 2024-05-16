for dataset in "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python3.8 ./linkan.py --input_path ./outputs/merged/ --suffix $dataset >> ./outputs/merged/output_linkan.txt
done

python3.8 ./gpt.py --base_dir ./outputs/merged/ >> ./outputs/merged/output_gpt.txt

python3.8 ./bert.py --base_dir ./outputs/merged/ >> ./outputs/merged/output_bert.txt
