dataset_path="./TIME/TIME_Newest.json"
model="model_path/Llama-3.1-8B-Instruct"

for dataset_name in "TIME-Lite-Wiki" "TIME-Lite-Dial"
do
    for task in "Extract" "Localization" "Computation" "Duration_Compare" "Order_Compare" "Explicit_Reasoning" "Order_Reasoning" "Relative_Reasoning" "Co_temporality" "Timeline"  "Counterfactual" 
    do
        echo "Evaluating ${dataset_name} with ${task} task for model ${model}"
        python -m evaluation.eval \
            --model_path ${model} \
            --dataset_path ${dataset_path} \
            --dataset_name ${dataset_name} \
            --task ${task}
    done
done

dataset_name="TIME-Lite-News"
for task in "Localization" "Computation" "Duration_Compare" "Order_Compare" "Explicit_Reasoning" "Order_Reasoning" "Relative_Reasoning" "Co_temporality" "Timeline"  "Counterfactual" 
do
    for retriever in "bm25" "vector" "hybrid"
    do
        echo "Evaluating ${dataset_name} with ${task} task for model ${model} and retriever ${retriever}"
        python -m evaluation.eval \
            --model_path ${model} \
            --dataset_path ${dataset_path} \
            --dataset_name ${dataset_name} \
            --task ${task} \
            --retriever ${retriever}
    done
done
