src=$1
tgt=$2

GPUS=$3
MODEL_NAME=$4
STEPS=$5

EXPDIR=$PWD
DATASET=$PWD/data
BIN_DATA=$DATASET/tmp/bin-data

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train $BIN_DATA -s ${src} -t ${tgt} \
		            --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
                    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
                    --dropout 0.3 --weight-decay 0.0001 \
                    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                    --max-tokens 4096 \
                    --patience $STEPS \
                    --eval-bleu \
                    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
                    --eval-bleu-detok moses \
                    --eval-bleu-remove-bpe \
                    --eval-bleu-print-samples \
                    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
					--save-dir $EXPDIR/models/$MODEL_NAME \
					2>&1 | tee $PWD/logs/${MODEL_NAME}

