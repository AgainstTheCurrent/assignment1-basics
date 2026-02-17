python scripts/train_transformer.py --vocab_size 10000 --context_length 256 --d_model 512  --d_ff 1344 --theta 10000 --num_layers 4 --num_heads 16 ^
                            --train_data ../data/TinyStoriesV2-GPT4-train-tokens.txt ^
                            --val_data ../data/TinyStoriesV2-GPT4-valid-tokens.txt ^
                            --batch_size 16 --lr 0.0001 --max_iters 10 --checkpoint_dir ../models/ --device cuda