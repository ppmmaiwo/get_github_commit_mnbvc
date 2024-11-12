# get_github_commit_mnbvc

###
### 安装
```shell
pip install -r requirements.txt
```


### 启动


```shell
usage: app.py [-h] [-i INPUT] [-t TOKENS] [-f] [-m MAX_SIZE]

Process some files.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the file containing repository list
  -t TOKENS, --tokens TOKENS
                        Path to the file containing GitHub tokens
  -f, --force           Force download all repositories
  -m MAX_SIZE, --max_size MAX_SIZE
                        Data file size settings: If it exceeds the maximum size, the file will be split. The default is 500 x 1024 x 1024 (500 MB).
  
```

参数说明

-i 指定 repos_list.txt

repos_list.txt 格式
```
369507628, https://github.com/1KomalRani/Komal.github.git
369507636, https://github.com/T1moB/RTS_Test_Game.git
369507639, https://github.com/MichaelGoldshmidt/MichaelGoldshmidt.github.io.git
369507643, https://github.com/LemonChocolate/Itwill_pdf.git
```
-t 指定 github_tokens.txt
github_tokens.txt 格式
```
token1
token2
token3
```

-f 强制下载 

* 指定 **-f** 参数 时，强制下载repos_list.txt
* 未指定 **-f** 参数 时，success.txt包含的记录不再重新下载

其他文件说明

* github_ips.txt github对应的ip列表
* success.txt 成功下载的repos_id
* fail.txt 下载失败的记录

。/data 成功下载jsonl文件示例数据