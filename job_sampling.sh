for dataset in "KnowledgeGraphsDataSet" "Yago-Wiki" "person" "restaurant"  "anatomy" "doremus" "SPIMBENCH_small-2019" "SPIMBENCH_large-2016" "KnowledgeGraphsDataSet" "Yago-Wiki"
do
    python3.8 ./linking.py --suffix $dataset >> ./outputs/output_linking.txt
done
