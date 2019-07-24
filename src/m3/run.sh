#!/bin/sh

function extract_feature() {
    python3 gen_imprs.py
    python3 gen_samples.py

    python3 feat0.py
    python3 gen_item_session_action_feat.py
    python3 gen_item_sid_clk_impr.py
    python3 gen_all_act.py
    python3 gen_ctr_feat.py
    python3 gen_item_last_act.py
    python3 gen_session_feat.py
    python3 gen_last_act_feat.py
    python3 gen_item_feat.py
    python3 feat12.py
    python3 feat13.py
    python3 gen_last_act_item_diff.py
    python3 gen_uid_item_last_act.py
    python3 gen_sid_impr_rank.py
    python3 gen_sid_item_neighbor.py
    python3 feat21.py
    python3 gen_top30_feat.py
    python3 gen_user_feat.py
    python3 gen_impr_list_feat.py
    python3 gen_impr_list_feat2.py
    python3 feat32.py
}


function merge_top_30_and_single_models() {
    python3 feat34.py
    
    cp ../m2/model/lgb_s0_m2_107/lgb_s0_m2_107.csv ../../model/te_m2_107.csv
    cp ../m2/model/lgb_s0_m2_107/lgb_s0_m2_107_cv.csv ../../model/tr_m2_107.csv
    cp ../m2/model/lgb_s0_m2_38/lgb_s0_m2_38.csv ../../model/te_m2_38.csv
    cp ../m2/model/lgb_s0_m2_38/lgb_s0_m2_38_cv.csv ../../model/tr_m2_38.csv
    cp ../m2/model/lgb_s0_m2_87/lgb_s0_m2_87.csv ../../model/te_m2_87.csv
    cp ../m2/model/lgb_s0_m2_87/lgb_s0_m2_87_cv.csv ../../model/tr_m2_87.csv
}


function train_71() {
    merge_top_30_and_single_models
    python3 feat71.py
    python3 lgb_cv.py 71 50 0.02
}


function train_72() {
    python3 feat72.py
    python3 lgb_cv.py 72 50 0.02
    
    cp ../../model/m3_72/sub.csv ../../output/final_sub.csv
}


case $1 in
    feat)
        extract_feature
        ;;
    model)
        train_71
        ;;
    final)
        train_72
        ;;
    ?) 
        echo "unkonw argument"
        exit 1
        ;;
esac


