#!/bin/bash

# 创建 getResult.py 输出的CSV文件，并写入表头
echo "i,f1_quatrain,MCC,TP,TN,FP,FN,+RECALL,-RECALL" > 6_threshold_results.csv

# 设置初始值为0.1，结束值为0.9，步长为0.1
for ((i=1; i<=9; i++)); do
    # 计算当前i对应的小数值
    #decimal_i=$(bc <<< "scale=1; $i/10")
    echo "Running getResult.py for threshold=$i"
    # 运行 getResult.py 并传入小数值
    python3 gmn/getResult.py ws_6 79 $i > temp_result.txt

    # 从临时文件中读取需要的指标，并写入CSV文件
    while IFS=: read -r _ f1_quatrain _ MCC _ tp _ tn _ fp _ fn _ pos_recall _ neg_recall; do
    	   echo "$i,$f1_quatrain,$MCC,$tp,$tn,$fp,$fn,$pos_recall,$neg_recall" >> 6_threhold_results.csv
    done < <(grep -E "f1_quatrain:|MCC:|TP:|TN:|FP:|FN:|\+RECALL:|\-RECALL:" temp_result.txt)

done

# 删除临时文件
rm temp_result.txt
