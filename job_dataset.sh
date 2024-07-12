for dataset in "KnowledgeGraphsDataSet" "Yago-Wiki" "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016"
do
    python3.8 ./dataset_stats.py --input_path ./inputs/ --suffix $dataset >> ./outputs/output_stats.txt
done