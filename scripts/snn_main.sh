python3.8 snn_main.py \
    --tabular_lr "'fit_one_cycle'" \
    --tabular_n_epoch "3" \
    --tabular_n_workers "12" \
    --tabular_batch_size "128" \
    --tabular_layer_dropout "0." \
    --tabular_embed_dropout "0." \
    --tabular_layer_1_neurons "2048" \
    --tabular_layer_2_neurons "1024" \
    --tabular_layer_3_neurons "128" \
    --snn_lr "10e-6" \
    --snn_n_out "32" \
    --snn_margin "0.2" \
    --snn_n_epoch "50" \
    --snn_n_workers "20" \
    --snn_batch_size "128" \
    --model_dir "'.'" \
    --device "'cpu'" \
    --sample "1.0"