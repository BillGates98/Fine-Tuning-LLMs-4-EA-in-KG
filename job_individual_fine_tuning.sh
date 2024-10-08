for dataset in "KnowledgeGraphsDataSet" "Yago-Wiki" "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python3.8 ./linkan.py --suffix $dataset >> ./outputs/output_linkan_vec.txt
    # python3.8 ./linkan.py --suffix $dataset >> ./outputs/output_linkan.txt
done

# python3.8 ./gpt.py --base_dir ./outputs/ --suffix no --enable_trainer False >> ./outputs/output_gpt.txt
# python3.8 ./bert.py --base_dir ./outputs/ --suffix no --enable_trainer False >> ./outputs/output_bert.txt
# # 
