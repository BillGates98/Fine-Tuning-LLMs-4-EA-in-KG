for dataset in "KnowledgeGraphsDataSet" # "Yago-Wiki" "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python3.8 ./linkan.py --input_path ./outputs/merged4/ --suffix $dataset >> ./outputs/merged4/output_linkan.txt
done

python3.8 ./gpt.py --base_dir ./outputs/merged4/ --suffix model_gpt2 --enable_trainer True # >> ./outputs/merged4/new_output_gpt.txt

python3.8 ./bert.py --base_dir ./outputs/merged4/ --suffix model_bert --enable_trainer True # >> ./outputs/merged4/new_output_bert.txt
