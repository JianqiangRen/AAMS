#!/usr/bin/env bash
python test.py --model tf_model/aams.pb \
                        --content images/content/lenna_cropped.jpg \
                        --style images/style/candy.jpg \
                        --inter_weight 1.0