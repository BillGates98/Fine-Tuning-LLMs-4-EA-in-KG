for dataset in "KnowledgeGraphsDataSet" "Yago-Wiki" "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python3.8 ./linkan.py --input_path ./outputs/merged/ --suffix $dataset >> ./outputs/merged/new_output_linkan_vec.txt
done

# python3.8 ./gpt.py --base_dir ./outputs/merged/ --suffix merged --enable_trainer True >> ./outputs/merged/new_output_gpt.txt

# python3.8 ./bert.py --base_dir ./outputs/merged/ --suffix merged --enable_trainer True >> ./outputs/merged/new_output_bert.txt
