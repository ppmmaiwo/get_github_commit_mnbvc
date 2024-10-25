import argparse
import concurrent.futures
import glob
import hashlib
import sys
import traceback
from urllib.parse import urlparse

import requests
import json
import os
from datetime import datetime
import time
from tqdm import tqdm
import logging
import re
import urllib3
from logging.handlers import RotatingFileHandler

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Setting up logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# 创建 logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置最低的日志级别为 DEBUG

# 创建 console handler，输出 DEBUG 及以上级别的日志到 console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # 输出 DEBUG 日志到 console

# 创建 file handler，输出 INFO 及以上级别的日志到文件，设置文件大小为 10M，最多保留 3 个文件
file_handler = RotatingFileHandler("app.log", maxBytes=10 * 1024 * 1024, backupCount=3)
file_handler.setLevel(logging.INFO)  # 输出 INFO 及以上级别的日志到文件

# 创建格式化器，设定日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 为 handlers 设置格式
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将 handlers 添加到 logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class Config:
    def __init__(self, repo_list_path, token_path, ips_path, success_path, fail_path, output_dir='./output'):
        self.repo_list = self.load_repos(repo_list_path)
        self.session = HTTPSessionManager.create_session()
        self.token_manager = GitHubTokenManager(token_path, ips_path, self.session)
        self.success_path = success_path
        self.output_dir = output_dir
        self.fail_path = fail_path
        self.success_repos = self.load_success()
        self.safe_delete_file(self.fail_path)
        logger.info("Config initialized.")

    def load_repos(self, path):
        with open(path, 'r') as file:
            repos = [line.strip().split(',') for line in file.readlines()]
        logger.debug(f"Loaded {len(repos)} repositories.")
        return repos

    def load_success(self):
        success = set()
        if os.path.exists(self.success_path):
            # 打开文件进行读取
            with open(self.success_path, 'r') as file:
                for line in file:
                    # 移除每行的换行符并添加到集合中
                    success.add(line.strip())
            logger.info("Loaded success records.")
            #
            with open(self.success_path, 'w') as file:
                for item in success:
                    # 写入每个字符串到文件，每个字符串占一行
                    file.write(item + "\n")
            return success
        return set()

    def mark_success(self, repo_id):
        self.success_repos.add(repo_id)
        with open(self.success_path, 'a') as file:
            # 写入一行内容，并添加换行符
            file.write(f"{repo_id}\n")
        logger.info(f"Marked {repo_id} as successfully processed.")

    def mark_fail(self, repo_id, repo_url):
        with open(self.fail_path, 'a') as file:
            # 写入一行内容，并添加换行符
            file.write(f"{repo_id},{repo_url}\n")
        logger.info(f"Marked {repo_id},{repo_url} as fail processed.")

    def clean_success(self):
        self.success_repos = set()
        self.safe_delete_file(config.success_path)

    def safe_delete_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    def get_header(self, url):
        token = self.token_manager.get_token()
        host = urlparse(url).hostname
        headers = {
            "Authorization": f"token {token}",
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36",
            "Connection": "close",
            "host": host
        }
        return headers


class GitHubTokenManager:
    RATE_LIMIT_URL = "https://api.github.com/rate_limit"

    def __init__(self, token_file, ips_path, session):
        self.tokens = self.load_tokens(token_file)
        self.ips = self.load_ips(ips_path)
        self.current_token_index = 0
        self.token_remaining = 0
        self.session = session
        self.current_ip_index = 0
        self.ips_speed = list()
        if len(self.tokens) == 0:
            logging.error("github_tokens.txt 需要至少要填入一个token")
            sys.exit(1)

    def test_speed(self, url="https://github.com"):
        # domain = "github.com"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_ip = {executor.submit(self.test_ip_speed, url, ip): ip for ip in self.ips}
            for future in concurrent.futures.as_completed(future_to_ip):
                speed = future.result()
                if speed['is_connected']:
                    self.ips_speed.append(speed)
        self.ips_speed.sort(key=lambda x: x['speed'])
        logging.debug("test speed result {}".format(self.ips_speed))
        # return speeds

    def test_ip_speed(self, url: str, ip: str):
        try:
            hostname = urlparse(url).hostname
            new_url = url.replace(hostname, ip, 1)
            logger.debug("Testing IP Speed {} for {} host {}".format(url, ip, hostname))
            r = requests.head(new_url, headers={"host": hostname}, verify=False, timeout=5)
            logger.debug("IP Speed {} for {} status_code {}".format(hostname, ip, r.status_code))
            if r.status_code < 500:
                return {'ip': ip, 'speed': r.elapsed.microseconds, 'is_connected': True}
            else:
                return {'ip': ip, 'speed': r.elapsed.microseconds, 'is_connected': False}
        except Exception as e:
            logger.error("test_ip_speed error {}".format(e))
            return {'ip': ip, 'speed': float('inf'), 'is_connected': False}

    def load_tokens(self, file_path):
        with open(file_path, 'r') as file:
            tokens = [line.strip() for line in file.readlines()]
        logger.debug(f"Loaded {len(tokens)} tokens.")
        return tokens

    def load_ips(self, file_path):
        with open(file_path, 'r') as file:
            ips = [line.strip() for line in file.readlines()]
        logger.debug(f"Loaded {len(ips)} ips.")
        return ips

    def get_token(self):
        while self.token_remaining <= 0:
            try:
                self.rotate_token()
                time.sleep(1)
            except Exception as e:
                logger.error("{} {}".format(traceback.format_exc(), e))

        return self.get_current_token()

    def get_url(self, url):
        host = urlparse(url).hostname
        return url.replace(host, self.get_current_ip(), 1)

    def get_current_ip(self):
        ip_index = self.current_ip_index % len(self.ips_speed)
        return self.ips_speed[ip_index]["ip"]

    def rotate_token(self):
        if self.token_remaining > 0:
            logger.debug(
                f"Rotating token. return current token.  {self.get_current_token()} {self.current_ip_index}  / {len(self.ips_speed)}.")
            return
        if self.token_remaining <= 0:
            if self.current_ip_index >= len(self.ips_speed) - 1:
                self.current_token_index += 1
                self.current_ip_index = 0
                logger.info(
                    f"Rotated  token   {self.get_current_token()} {self.current_ip_index}  / {len(self.ips_speed)}.")
            else:
                self.current_ip_index += 1
                logger.info(
                    f"Rotated  ip   {self.get_current_token()} {self.current_ip_index}  / {len(self.ips_speed)}.")

        response = self.request_rate_limit()
        if response is None:
            self.token_remaining = 0
            return

        limit = response.headers.get("X-RateLimit-Limit")
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_time = response.headers.get("X-RateLimit-Reset")
        current_timestamp = int(time.time())
        time_diff = int(reset_time) - current_timestamp + 1
        logging.info(
            f"{self.RATE_LIMIT_URL} 可请求次数: {limit} 剩余请求次数: {remaining} token_remaining {self.token_remaining} 重置时间还有: {time_diff // 60} 分钟")
        if self.token_remaining > 100:
            logger.info(f"Token has sufficient limit: {self.token_remaining} remaining.")
        else:
            # 所有token试过之后，才等待重置
            if self.current_token_index < len(self.tokens):
                logger.warning(f"Token limit is low: {self.token_remaining} remaining. rotate to next.")
                self.token_remaining = 0
                return
            else:
                logger.info("所有token都尝试过了，开始等待reset")
                logger.info(f"wait token {self.get_current_token()} to reset")
                self.wait_for_reset(time_diff)
                response = self.session.get(self.RATE_LIMIT_URL,
                                            headers={"Authorization": f"token {self.get_current_token()}"})
                self.token_remaining = response.json()["resources"]["core"]["remaining"]
                return

    def request_rate_limit(self):
        try:
            host = urlparse(self.RATE_LIMIT_URL).hostname
            response = self.session.get(self.get_url(self.RATE_LIMIT_URL),
                                        headers={"Authorization": f"token {self.get_current_token()}", "host": host},
                                        verify=False, timeout=5 * 60)
            self.token_remaining = response.json()["resources"]["core"]["remaining"]
        except Exception as e:
            logger.error(e)
            return None
        return response

    def get_current_token(self):
        token_index = self.current_token_index % len(self.tokens)
        return self.tokens[token_index]

    def consume_token(self):
        self.token_remaining -= 1

    def update_token_remaining(self, remaining):
        if remaining is None:
            return
        self.token_remaining = int(remaining)
        logger.debug(f"Token  {self.get_current_token()} update token_remaining: {self.token_remaining}")

    def wait_for_reset(self, wait_time):
        logger.info(f"Waiting for rate limit reset: {wait_time} seconds.")
        with tqdm(total=wait_time, desc="Waiting for token reset") as pbar:
            while wait_time > 0:
                wait_time -= 1
                time.sleep(1)
                pbar.update(1)
        logger.info("Token reset complete. Continuing operations.")


class GitHubRepository:

    def __init__(self, repo_url, config):
        self.MAX_FILE_SIZE = 500 * 1024 * 1024
        self.repo_url = repo_url
        self.config = config
        self.token_manager = config.token_manager
        self.session = requests.Session()
        logger.info(f"Initialized repository: {repo_url}")

    def get_github_repos_info(self, url):
        match = re.search(r"github\.com/([^/]+)/([^/]+)\.git", url)
        if not match:
            logger.error(f"Could not find github commit url ${url}")
            return None
        username = match.group(1)
        repository = match.group(2)
        return username, repository

    def get_github_commit_api_url(self, url):
        username, repository = self.get_github_repos_info(url)
        return f"https://api.github.com/repos/{username}/{repository}/commits"

    def get_github_commit_details_api_url(self, url, sha):
        username, repository = self.get_github_repos_info(url)
        return f"https://api.github.com/repos/{username}/{repository}/commits/{sha}"

    def get_github_compare_api_url(self, url, parent_sha, sha):
        username, repository = self.get_github_repos_info(url)
        return f"https://api.github.com/repos/{username}/{repository}/compare/{parent_sha}...{sha}"

    def get_commits(self, repo_id, repo_url):
        page = 1
        commits = []
        while True:
            params = {"page": page, 'per_page': 100}
            commits_url = self.get_github_commit_api_url(repo_url)
            logger.debug(f"get_commit:  {commits_url}  page: {page}")
            headers = self.config.get_header(commits_url)
            self.config.token_manager.consume_token()
            response = self.config.session.get(self.config.token_manager.get_url(commits_url), params=params,
                                               headers=headers, verify=False, timeout=5 * 60)
            remaining = response.headers.get("X-RateLimit-Remaining")
            self.config.token_manager.update_token_remaining(remaining)
            if response.status_code == 200:
                commits_data = response.json()
                if len(commits_data) == 0:
                    logging.debug("没有更多提交了，退出循环")
                    break  # 没有更多提交了，退出循环
                commits.extend(commits_data)
                page += 1
            else:
                logger.error(f"Failed to fetch page {page} for {repo_id}/{repo_url}: {response.status_code}")
                return None
        logger.info(f"Fetched {len(commits)} commits for {repo_id}/{repo_url}")
        return commits

    def process_commit(self, commit, repo_id, repo_url):
        commit_jsonl_data=[]
        commit_sha = commit["sha"]
        commit_details_url = self.get_github_commit_details_api_url(repo_url, commit_sha)
        logger.debug("get commit_details_url:{}".format(commit_details_url))
        headers = self.config.get_header(commit_details_url)
        self.config.token_manager.consume_token()
        commit_details_response = self.session.get(self.config.token_manager.get_url(commit_details_url), headers=headers, verify=False,
                                         timeout=5 * 60)
        username, repository = self.get_github_repos_info(repo_url)
        remaining = commit_details_response.headers.get("X-RateLimit-Remaining")
        self.config.token_manager.update_token_remaining(remaining)
        if commit_details_response.status_code == 200:
            commit_details = commit_details_response.json()
            logger.debug("get commit_details_url:{} sucess".format(commit_details_url))
            # 获取 diff 内容
            for file in commit_details.get("files", []):
                file_name = file.get("filename")
                file_extension = file_name.split(".")[-1] if "." in file_name else ""
                diff_content = file.get("patch", "")
                file_path = file.get("filename")
                diff_md5 = hashlib.md5(diff_content.encode('utf-8')).hexdigest() if diff_content else ""
                commit_date = commit_details["commit"]["author"]["date"]
                formatted_date = datetime.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d")

                jsonl_data = {
                    "来源": "github",
                    "仓库名": f"{username}/{repository}",
                    "path": file_path,
                    "文件名": file_name.split("/")[-1],
                    "ext": file_extension,
                    "index": f"{commit_details['parents'][0]['sha']}..{commit_details['sha']}" if commit_details.get(
                        'parents') else commit_details["sha"],
                    "message": commit_details["commit"]["message"],
                    "diff": diff_content,
                    "原始编码": "GBK",
                    "md5": diff_md5,
                    "时间": formatted_date,
                    "扩展字段": json.dumps({})
                }
                commit_jsonl_data.append(jsonl_data)
            return commit_jsonl_data
        else:
            logger.debug(commit_details_response.json())
            logger.error(f"Failed to fetch diff data for commit {commit_sha}.")
            return None

    def write_to_file(self, json_data, filename):
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        with open(filename, 'a', encoding='utf-8') as jsonl_file:
            jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")

    def get_next_filename(self, base_filename):
        """
            Generate the next filename based on existing files
        """
        i = 1
        while True:
            if i == 1:
                new_filename = os.path.join(self.config.output_dir, f'{base_filename}.jsonl')
            else:
                new_filename = os.path.join(self.config.output_dir, f'{base_filename}_{i}.jsonl')

            if not os.path.exists(new_filename):
                return new_filename
            else:
                file_size = os.path.getsize(new_filename)
                if file_size < self.MAX_FILE_SIZE:
                    return new_filename
            i += 1

    def remove_all_jsonl_files(self, base_filename):
        self.config.safe_delete_file(os.path.join(self.config.output_dir, f'{base_filename}.jsonl'))

        files = glob.glob(os.path.join(self.config.output_dir, f'{base_filename}_*.jsonl'))
        # 删除每个匹配的文件
        for file in files:
            try:
                os.remove(file)
                logger.info(f"已删除文件: {file}")
            except OSError as e:
                logger.info(f"删除文件 {file} 时出错: {e}")

    def getout_file(self, filename):
        # return os.path.join(self.config.output_dir, f"{filename}-commit.jsonl")
        return self.get_next_filename(filename)


class HTTPSessionManager:
    @staticmethod
    def create_session():
        """
        Create and configure a requests session with custom retry logic and connection settings.
        """
        requests.adapters.DEFAULT_RETRIES = 30
        s = requests.session()
        s.keep_alive = False
        return s


class Application:
    def __init__(self, config):
        self.config = config
        logger.info("Application started.")

    def test_tokens(self):
        logger.info("test tokens")
        config.token_manager.RATE_LIMIT_URL = "http://127.0.0.1"
        config.token_manager.ips = ['127.0.0.1','localhost']
        self.config.token_manager.test_speed("http://localhost")
        for i in range(50):
            logger.debug(f"test token loop {i}")
            logger.debug(
                f"test token  loop end {i} header {self.config.get_header(config.token_manager.RATE_LIMIT_URL)}")
            self.config.token_manager.update_token_remaining(0)

    def run(self):
        self.config.token_manager.test_speed()
        for repo_id, repo_url in self.config.repo_list:
            if repo_id.strip() in self.config.success_repos:
                logger.info(f"Skipping {repo_id} as it's already processed.")
                continue
            repo = GitHubRepository(repo_url.strip(), self.config)
            try:
                self.config.safe_delete_file(repo.getout_file(repo_id))
                commits = repo.get_commits(repo_id, repo_url)
                num_commit = len(commits)
                for k, commit in tqdm(enumerate(commits), total=num_commit, desc="Processing commit",
                                      unit="commit"):
                    commit_data = repo.process_commit(commit, repo_id, repo_url)
                    repo.write_to_file(commit_data, repo.getout_file(repo_id))
                self.config.mark_success(repo_id)
            except Exception as e:
                self.config.safe_delete_file(repo.getout_file(repo_id))
                self.config.mark_fail(repo_id, repo_url)
                logger.exception("crawl {} exception {}  {}".format(repo_url, traceback.format_exc(), e))


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process some files.")

    # 添加 -i 参数，指定类型为文件类型，这样可以检查文件是否存在
    parser.add_argument('-i', '--input', type=str, default='repos_list.txt',
                        help='Path to the file containing repository list')

    # 添加 -t 参数，同样指定类型为文件类型
    parser.add_argument('-t', '--tokens', type=str, default='github_tokens.txt',
                        help='Path to the file containing GitHub tokens')

    # 添加 -i 参数，不强制，只判断有没有
    parser.add_argument('-f', '--force', action='store_true', help='Force download all repositories')

    # 解析命令行参数
    args = parser.parse_args()

    config = Config(args.input, args.tokens, "github_ips.txt", 'success.txt', 'fail.txt')
    if args.force:
        config.clean_success()
    app = Application(config)
    app.run()
    # app.test_tokens()
