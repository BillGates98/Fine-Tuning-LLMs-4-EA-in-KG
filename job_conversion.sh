# #!/bin/bash
# while IFS=',,,' read -r dataset sourcef targetf validsameas; do
#     echo "Dataset truths : $line will converted ...0%"
#     mv ./validations/$dataset/refalign.rdf ./validations/$dataset/refalign.xml
#     python3.8 ./convert_xml_to_graph.py --input_data ./inputs/$dataset/refalign.xml --input_path ./inputs/$dataset/ --output_path ./inputs/$dataset/
#     echo "File : $line will converted ...100%"
# done < "$1"


# python3.8 ./convert_xml_to_graph.py --input_data ./inputs/Yago-Wiki/reference.rdf --input_path ./inputs/Yago-Wiki/ --output_path ./inputs/Yago-Wiki/
python3.8 ./convert_xml_to_graph.py --input_data ./inputs/KnowledgeGraphsDataSet/reference.rdf --input_path ./inputs/KnowledgeGraphsDataSet/ --output_path ./inputs/KnowledgeGraphsDataSet/
    