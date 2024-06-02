#!/bin/bash

# ���� getResult.py �ű�
# ���� i ��ȡֵ��ΧΪ 0 �� 99
# ������� getResult.py �ű���i ��ֵ�� 0 �� 99

# ����һ�����ڱ��������ļ�����д���ͷ
echo "i,f1_quatrain,auc_quatrain" > BATS8_10_test_results.csv

# ���� i ��ȡֵ��Χ
for i in {1..99}; do
    echo "Running getResult.py for i=$i"
    # ���� getResult.py �ű��������������ʱ�ļ�
    python3 gmn/getResult.py vsBATS8_10 $i > temp_result.txt

    # ����ʱ�ļ�����ȡ f1_quatrain �� auc_quatrain����д�����ļ�
    while IFS=: read -r _ f1_quatrain _ auc_quatrain; do
        echo "$i,$f1_quatrain,$auc_quatrain" >> BATS8_10_test_results.csv
    done < <(grep -E "f1_quatrain:|auc_quatrain:" temp_result.txt)
done

# ɾ����ʱ�ļ�
rm temp_result.txt
