for dataset in "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python ./linkan.py --input_path ./outputs/merged/ --suffix $dataset >> ./outputs/merged/output_linkan.txt
done

python ./gpt.py --base_dir ./outputs/merged/ --enable_trainer True >> ./outputs/merged/output_gpt.txt

python ./bert.py --base_dir ./outputs/merged/ --enable_trainer True >> ./outputs/merged/output_bert.txt
