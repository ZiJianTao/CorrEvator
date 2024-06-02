@echo off
rem 运行 getResult.py 脚本
rem 设置 i 的取值范围为 0 到 99
rem 逐个运行 getResult.py 脚本，i 的值从 0 到 99

rem 创建一个用于保存结果的文件，并写入表头
echo i,f1_quatrain,auc_quatrain > test_results.csv

rem 遍历 i 的取值范围
for /L %%i in (20,1,99) do (
    echo Running getResult.py for i=%%i
    rem 运行 getResult.py 脚本，将结果存入临时文件
    python gmn/getResult.py remove_5 %%i > temp_result.txt

    rem 从临时文件中提取 f1_quatrain 和 auc_quatrain，并写入结果文件
    for /F "tokens=2,4 delims=:" %%a in ('type temp_result.txt ^| findstr /C:"f1_quatrain:" /C:"auc_quatrain:"') do echo %%i,%%a,%%b >> test_results.csv
)

rem 删除临时文件
del temp_result.txt
