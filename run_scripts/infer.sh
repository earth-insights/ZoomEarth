mdoel_name=""
exp_name=""
echo "Infering model: $model_name on LRS-GRO!"
echo "Experiment name: $exp_name!"
python src/infer.py \
    --model_name $model_name \
    --exp_name $exp_name