export CUDA_VISIBLE_DEVICES=$1

#cd ..

case $2 in
  mse|dilate|tildeq)
  loss=$2
  ;;
  *)
  echo "Please choose one of three loss functions (mse, dilate, tildeq)"
  exit
  ;;
esac

case $3 in
  Autoformer|Transformer|Informer|Reformer|nbeats|FEDformer|ns_Transformer)
  model=$3
  ;;
  *)
  echo "Please choose one of available models from Autoformer, Transformer, Informer, Reformer, nbeats, FEDformer, and ns_Transformer"
  exit
  ;;
esac

for preLen in 96 192 336 720
do

# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTm2.csv \
  --model $model \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des $loss \
  --d_model 512 \
  --itr 3 \
  --loss $loss \
  --model_id 'ETTm2_96_'$preLen \
  > 'nohup_'$model'_ETTm2_'$preLen'_'$loss'.out'

# ETTh2
python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTh2.csv \
  --model $model \
  --data ETTh2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des $loss \
  --d_model 512 \
  --itr 3 \
  --loss $loss \
  --model_id 'ETTh2_96_'$preLen \
  > 'nohup_'$model'_ETTh2_'$preLen'_'$loss'.out'


## electricity
python -u run.py \
 --is_training 1 \
 --root_path ./data/electricity/ \
 --data_path electricity.csv \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des $loss \
 --itr 3 \
 --loss $loss \
 --model_id 'ECL_96_'$preLen \
 > 'nohup_'$model'_ECL_'$preLen'_'$loss'.out'


# exchange
python -u run.py \
 --is_training 1 \
 --root_path ./data/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des $loss \
 --itr 3 \
 --loss $loss \
 --model_id 'Exchange_96_'$preLen \
 > 'nohup_'$model'_Exchange_'$preLen'_'$loss'.out'


# traffic
python -u run.py \
 --is_training 1 \
 --root_path ./data/traffic/ \
 --data_path traffic.csv \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des $loss \
 --itr 3 \
 --train_epochs 3 \
 --loss $loss \
 --model_id 'Traffic_96_'$preLen \
 > 'nohup_'$model'_traffic_'$preLen'_'$loss'.out'


# weather
python -u run.py \
 --is_training 1 \
 --root_path ./data/weather/ \
 --data_path weather.csv \
 --model $model \
 --data custom \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des $loss \
 --itr 3 \
 --loss $loss \
 --alpha 0.2 \
 --model_id 'Weather_96_'$preLen \
 > 'nohup_'$model'_weather_'$preLen'_'$loss'.out'

done

done

done

