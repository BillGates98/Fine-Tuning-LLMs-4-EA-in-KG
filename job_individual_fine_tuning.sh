for dataset in "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python ./linkan.py --suffix $dataset >> ./outputs/output_linkan.txt
done

# job_individual_fine_tuning.sh

python ./gpt.py --base_dir ./outputs/ >> ./outputs/output_gpt.txt
python ./bert.py --base_dir ./outputs/ >> ./outputs/output_bert.txt
